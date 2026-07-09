from __future__ import annotations
from pathlib import Path
from typing import Protocol, runtime_checkable
from ..config import Config, models_dir
from .text_embed import resolve_device


@runtime_checkable
class ImageEmbedder(Protocol):
    @property
    def dim(self) -> int: ...

    def encode_image(self, paths: list[str]) -> list[list[float] | None]: ...

    def encode_text(self, texts: list[str]) -> list[list[float]]: ...


class OpenClipEmbedder:
    def __init__(self, model_name: str, pretrained: str, device: str = "cpu", batch_size: int = 16):
        self.model_name = model_name
        self.pretrained = pretrained
        self._device_pref = device
        self.batch_size = batch_size
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            import open_clip

            device = resolve_device(self._device_pref)
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained, cache_dir=str(models_dir())
            )
            model.eval()
            model.to(device)
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._device = device
        return self._model

    @property
    def device(self) -> str:
        return resolve_device(self._device_pref)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self.encode_text(["probe"])
        return self._dim

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import torch

        model = self._load()
        toks = self._tokenizer(texts).to(self._device)
        with torch.no_grad():
            feats = model.encode_text(toks)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        self._dim = feats.shape[1]
        return feats.cpu().tolist()

    def encode_image(self, paths: list[str]) -> list[list[float] | None]:
        if not paths:
            return []
        import torch
        from PIL import Image

        model = self._load()
        out: list[list[float] | None] = [None] * len(paths)
        batch_idx: list[int] = []
        tensors = []
        results: dict[int, list[float]] = {}

        def flush():
            if not tensors:
                return
            stack = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                feats = model.encode_image(stack)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            self._dim = feats.shape[1]
            for j, idx in enumerate(batch_idx):
                results[idx] = feats[j].cpu().tolist()
            tensors.clear()
            batch_idx.clear()

        for i, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(self._preprocess(img))
                batch_idx.append(i)
            except Exception:
                continue
            if len(tensors) >= self.batch_size:
                flush()
        flush()
        for idx, vec in results.items():
            out[idx] = vec
        return out


def get_image_embedder(cfg: Config) -> ImageEmbedder:
    return OpenClipEmbedder(cfg.image_model, cfg.image_pretrained, device=cfg.device)


def _stem_text(path: str) -> str:
    stem = Path(path).stem
    return "".join((c if c.isalnum() else " " for c in stem)).strip()
