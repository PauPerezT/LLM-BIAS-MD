# MentaLLaMA Classification Pipeline

A clean, repo-ready PyTorch Lightning pipeline for classifying mental health labels using a frozen MentaLLaMA backbone with a lightweight classifier head.

## Features
- Frozen `klyang/MentaLLaMA-chat-7B-hf` backbone with masked mean pooling
- TorchMetrics (accuracy, macro recall) + sklearn UAR
- Class-weighted cross-entropy (optional, if dataset exposes `.weight`)
- Deterministic training & safe cleanup between runs
- W&B logging (optional; auto-disabled if `WANDB_API_KEY` not set)
- Config via `configs/config.yaml` (override with `CONFIG_PATH` env var)
- Predictions exported after test to `predictions_test.csv`

## Installation
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start
1. Implement your dataset adapter in `src/llama_pipeline/data.py` (see **Dataset contract** below).
2. (Optional) Set tokens:
```bash
export HF_TOKEN=...           # if the HF model requires auth
export WANDB_API_KEY=...      # to enable Weights & Biases logging
```
3. (Optional) adjust `configs/config.yaml`.
4. Train/Test:
```bash
python -m llama_pipeline.refactored_llama_pipeline
```
Or:
```bash
bash scripts/train.sh
```

## Dataset contract
`get_dataset(csv_paths, audio_root_paths, LG: bool, lge: list[str])` must return a `torch.utils.data.Dataset` producing batches with keys:
- `input_ids: LongTensor[B, T]`
- `attention_mask: LongTensor[B, T]`
- `labels: LongTensor[B]`
- `age: LongTensor[B]`
- `gender: LongTensor[B]`
- `language: list[str]` (or tensor convertible to strings)
- Optional attribute `.weight: Tensor[num_labels]` for class weighting

If you currently return `{'values': Tensor, 'label': int}`, adapt your collate or dataset to this schema.

## Config
See `configs/config.yaml`. You can also override via environment variable:
```bash
export CONFIG_PATH=/absolute/path/to/your_config.yaml
```

## Notes
- The model is frozen for stability on small datasets. Unfreeze layers if you have more data.
- For multi-GPU or CPU, edit the `accelerator`/`devices` fields in the config.
- Predictions are saved to `save_predictions_dir/predictions_test.csv` after `trainer.test()`.

## License
MIT