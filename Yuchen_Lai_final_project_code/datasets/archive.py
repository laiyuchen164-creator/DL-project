from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import zipfile

import torch


class ZipExtractor:
    def __init__(self, zip_path: str | Path, cache_dir: str | Path) -> None:
        self.zip_path = Path(zip_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, member_name: str) -> Path:
        destination = self.cache_dir / member_name
        if destination.exists():
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.zip_path) as archive:
            archive.extract(member_name, path=self.cache_dir)
        return destination


def load_tensor_cache(path: str | Path) -> Optional[dict]:
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    return torch.load(cache_path, map_location="cpu")


def save_tensor_cache(path: str | Path, payload: dict) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temp_path)
    os.replace(temp_path, cache_path)
