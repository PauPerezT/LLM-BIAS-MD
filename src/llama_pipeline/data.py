"""
Dataset adapter stub.

Implement:
    get_dataset(csv_paths, audio_root_paths, LG: bool, lge: list[str]) -> torch.utils.data.Dataset

Expected batch item keys:
  - 'input_ids': LongTensor[B, T]
  - 'attention_mask': LongTensor[B, T]
  - 'labels': LongTensor[B]
  - 'age': LongTensor[B]
  - 'gender': LongTensor[B]
  - 'language': list[str] or tensor convertible to strings

Optionally add attribute `.weight` (Tensor[num_labels]).
"""
from typing import Any, List, Sequence, Union
import torch
from torch.utils.data import Dataset

def get_dataset(csv_paths: Union[str, Sequence[str]],
                audio_root_paths: Union[str, Sequence[str]],
                LG: bool,
                lge: List[str]) -> Dataset:
    raise NotImplementedError("Please implement get_dataset() according to the contract in README.md")