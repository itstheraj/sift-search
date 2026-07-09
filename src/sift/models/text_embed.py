from __future__ import annotations
import struct
from typing import Protocol, runtime_checkable
from ..config import Config, models_dir


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def encode(self, texts: list[str]) -> list[list[float]]: ...


def serialize(vec) -> bytes:
    return struct.pack(f"{len(vec)}f", *(float(x) for x in vec))


def resolve_device(pref: str) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    gpu = torch.cuda.is_available()
    if pref in ("auto", "rocm"):
        return "cuda" if gpu else "cpu"
    if pref == "cpu":
        return "cpu"
    if pref == "vulkan":
        return "cpu"
    return "cuda" if gpu else "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32):
        self.model_name = model_name
        self._device_pref = device
        self.batch_size = batch_size
        self._model = None
        self._dim: int | None = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = resolve_device(self._device_pref)
            self._model = SentenceTransformer(
                self.model_name, device=device, cache_folder=str(models_dir())
            )
            get_dim = (
                getattr(self._model, "get_embedding_dimension", None)
                or self._model.get_sentence_embedding_dimension
            )
            self._dim = int(get_dim())
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        return self._dim

    @property
    def device(self) -> str:
        return resolve_device(self._device_pref)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vecs = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()


def get_embedder(cfg: Config) -> Embedder:
    return SentenceTransformerEmbedder(cfg.text_model, device=cfg.device)
