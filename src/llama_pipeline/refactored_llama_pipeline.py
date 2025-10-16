from __future__ import annotations

import gc
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torchmetrics import Accuracy, Recall
from transformers import AutoModel
from sklearn.metrics import precision_recall_fscore_support
import yaml

from .data import get_dataset

@dataclass
class Config:
    seed: int = 42
    base_path_init: str = ""
    output_dir: str = "./checkpoints"
    project_name: str = "2C_LM_IS24"
    model_name: str = "klyang/MentaLLaMA-chat-7B-hf"
    max_epochs: int = 80
    patience: int = 40
    batch_size: int = 1
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-1
    accelerator: str = "gpu"
    devices: int | List[int] | str = 1
    use_wandb: bool = True
    save_predictions_dir: str = "."

def load_config() -> Config:
    cfg = Config()
    cfg_path = os.getenv("CONFIG_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "config.yaml"))
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            data = yaml.safe_load(f) or {}
        for k, v in (data or {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg

def metrics_sk(preds: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    class_names = ["Control", "MDD"]
    _, rec, _, _ = precision_recall_fscore_support(labels, preds, average=None, labels=np.array([0, 1]))
    out = {cls: float(r) for cls, r in zip(class_names, rec)}
    out["UAR"] = float(np.mean(rec))
    return out

def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom

class LLaMAClassifier(pl.LightningModule):
    def __init__(self, config: Config, num_labels: int = 2, class_weights: Optional[torch.Tensor] = None, hf_token: Optional[str] = None):
        super().__init__()
        self.config = config
        self.save_hyperparameters({"num_labels": num_labels, "model_name": config.model_name})

        self.llama = AutoModel.from_pretrained(config.model_name, token=hf_token, output_hidden_states=True)
        for p in self.llama.parameters():
            p.requires_grad = False

        hidden = self.llama.config.hidden_size
        self.proj = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, num_labels),
        )

        task = "binary" if num_labels == 2 else "multiclass"
        self.acc = Accuracy(task=task, num_classes=num_labels)
        self.recall_macro = Recall(task=task, average="macro", num_classes=num_labels)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self._preds: List[np.ndarray] = []
        self._labels: List[np.ndarray] = []
        self._scores: List[np.ndarray] = []
        self._ages: List[np.ndarray] = []
        self._genders: List[np.ndarray] = []
        self._langs: List[str] = []

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out = self.llama(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
        pooled = masked_mean_pool(out, batch["attention_mask"])
        logits = self.proj(pooled)
        return {"logits": logits}

    def _common_step(self, batch: Dict[str, Any], stage: str):
        logits = self(batch)["logits"]
        labels = batch["labels"]
        loss = self.criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage != "test"))
        self.acc.update(preds, labels)
        self.recall_macro.update(preds, labels)

        if stage != "train":
            self._preds.append(preds.detach().cpu().numpy())
            self._labels.append(labels.detach().cpu().numpy())
            self._scores.append(probs[:, 1].detach().cpu().numpy())
            self._ages.append(batch["age"].detach().cpu().numpy())
            self._genders.append(batch["gender"].detach().cpu().numpy())
            langs = batch["language"]
            if isinstance(langs, list):
                self._langs.extend([str(x) for x in langs])
            else:
                self._langs.extend([str(int(x)) for x in langs.detach().cpu().numpy()])
        return loss

    def training_step(self, batch, batch_idx): return self._common_step(batch, "train")
    def validation_step(self, batch, batch_idx): self._common_step(batch, "val")
    def test_step(self, batch, batch_idx): self._common_step(batch, "test")

    def on_validation_epoch_end(self):
        acc = self.acc.compute(); rec = self.recall_macro.compute()
        self.log("val_acc", acc, prog_bar=True); self.log("val_recall_macro", rec)
        self.acc.reset(); self.recall_macro.reset()
        if self._preds:
            preds, scores, labels = np.hstack(self._preds), np.hstack(self._scores), np.hstack(self._labels)
            for k, v in metrics_sk(preds, scores, labels).items():
                self.log(f"{k}_val", v)
        self._clear_buffers(keep_for_test=True)

    def on_test_epoch_end(self):
        acc = self.acc.compute(); rec = self.recall_macro.compute()
        self.log("test_acc", acc); self.log("test_recall_macro", rec)
        self.acc.reset(); self.recall_macro.reset()
        if self._preds:
            preds, scores, labels = np.hstack(self._preds), np.hstack(self._scores), np.hstack(self._labels)
            for k, v in metrics_sk(preds, scores, labels).items():
                self.log(f"{k}_test", v)
        self._export_predictions()
        self._clear_buffers(keep_for_test=False)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

    def _clear_buffers(self, keep_for_test: bool):
        # keep langs/ages/genders across val -> test if requested
        if not keep_for_test:
            self._preds.clear(); self._labels.clear(); self._scores.clear()
            self._ages.clear(); self._genders.clear(); self._langs.clear()

    def _export_predictions(self):
        if not self._preds: return
        preds = np.hstack(self._preds); labels = np.hstack(self._labels)
        ages = np.hstack(self._ages); genders = np.hstack(self._genders); languages = self._langs
        df = pd.DataFrame({"preds": preds, "targets": labels, "age": ages, "gender": genders, "language": languages})
        os.makedirs(self.config.save_predictions_dir, exist_ok=True)
        out_csv = os.path.join(self.config.save_predictions_dir, "predictions_test.csv")
        df.to_csv(out_csv, index=False)
        self.print(f"Saved test predictions to {out_csv}")

def build_loaders(set_name: str, dataset_name: str, lg_label: str, base_path_init: str, config: Config) -> tuple[DataLoader, DataLoader, DataLoader, Optional[torch.Tensor]]:
    path_set = os.path.join(base_path_init, set_name)
    if dataset_name == "AllLG":
        datasets = ["RADAR-MDD-CIBER-s1", "RADAR-MDD-KCL-s1", "RADAR-MDD-VUmc-s1"]
        csv_train = [os.path.join(path_set, f"{d}_train.csv") for d in datasets]
        csv_val = [os.path.join(path_set, f"{d}_val.csv") for d in datasets]
        csv_test = [os.path.join(path_set, f"{d}_test.csv") for d in datasets]
        roots = [f"/{d}/" for d in datasets]
        lge = ["ES", "EN", "NL"]
        train_ds = get_dataset(csv_train, roots, LG=True, lge=lge)
        val_ds = get_dataset(csv_val, roots, LG=True, lge=lge)
        test_ds = get_dataset(csv_test, roots, LG=True, lge=lge)
    else:
        root = f"/{dataset_name}/"
        train_ds = get_dataset(os.path.join(path_set, f"{dataset_name}_train.csv"), root, LG=False, lge=[lg_label])
        val_ds = get_dataset(os.path.join(path_set, f"{dataset_name}_val.csv"), root, LG=False, lge=[lg_label])
        test_ds = get_dataset(os.path.join(path_set, f"{dataset_name}_test.csv"), root, LG=False, lge=[lg_label])

    class_weights = getattr(train_ds, "weight", None)
    if not isinstance(class_weights, torch.Tensor): class_weights = None
    else: class_weights = class_weights.to(torch.float32)

    common = dict(batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    eval_common = dict(batch_size=1, shuffle=False, drop_last=False, num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, **eval_common)
    test_loader = DataLoader(test_ds, **eval_common)
    return train_loader, val_loader, test_loader, class_weights

def run_one_experiment(set_name: str, dataset_name: str, lg_label: str, hf_token: Optional[str], config: Config):
    train_loader, val_loader, test_loader, class_weights = build_loaders(set_name, dataset_name, lg_label, config.base_path_init, config)

    name = f"2Class_MentaLLaMA_{lg_label}_{set_name}"
    arch = "LLaMA_TextGen"

    os.makedirs(config.output_dir, exist_ok=True)
    ckpt = ModelCheckpoint(monitor="val_loss", dirpath=config.output_dir, filename=f"{arch}_{name}", save_top_k=1, mode="min")
    early = EarlyStopping(monitor="val_loss", patience=config.patience, mode="min")

    wandb_logger = None
    if config.use_wandb and os.getenv("WANDB_API_KEY"):
        wandb_logger = WandbLogger(log_model=False, project=config.project_name, name=name, config={
            "epochs": config.max_epochs, "batch_size": config.batch_size, "lr": config.lr,
            "dataset": f"{dataset_name}_{lg_label}", "architecture": arch,
        })

    model = LLaMAClassifier(config=config, num_labels=2, class_weights=class_weights, hf_token=hf_token)
    trainer = Trainer(enable_model_summary=True, max_epochs=config.max_epochs, callbacks=[ckpt, early], logger=wandb_logger,
                      accelerator=config.accelerator, devices=config.devices, gradient_clip_val=1.0, log_every_n_steps=10)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, dataloaders=test_loader)

    del trainer, model, train_loader, val_loader, test_loader
    gc.collect(); torch.cuda.empty_cache()

def main():
    config = load_config()
    seed_everything(config.seed, workers=True)
    hf_token = os.getenv("HF_TOKEN", None)

    sets = ["Age", "Classes", "Gender", "AGC"]
    datasets = ["RADAR-MDD-CIBER-s1", "RADAR-MDD-KCL-s1", "RADAR-MDD-VUmc-s1", "AllLG"]
    languages = ["ES", "EN", "NL", "All"]

    for set_name, dataset_name, lg_label in zip(sets, datasets, languages):
        run_one_experiment(set_name, dataset_name, lg_label, hf_token=hf_token, config=config)

if __name__ == "__main__":
    main()