from __future__ import annotations
import sys
from . import config, db, search


def _warn(msg: str) -> None:
    print(f"sift: {msg}", file=sys.stderr)


def build_embedder(cfg: config.Config):
    if not cfg.features.semantic:
        return None
    try:
        from .models.text_embed import get_embedder

        return get_embedder(cfg)
    except Exception as e:
        _warn(f"semantic disabled ({e})")
        return None


def build_image_embedder(cfg: config.Config):
    if not cfg.features.image:
        return None
    try:
        from .models.image_embed import get_image_embedder

        return get_image_embedder(cfg)
    except Exception as e:
        _warn(f"image search disabled ({e})")
        return None


def build_transcriber(cfg: config.Config):
    if not cfg.features.asr:
        return None
    try:
        from .models.asr import get_transcriber

        return get_transcriber(cfg)
    except Exception as e:
        _warn(f"transcription disabled ({e})")
        return None


def build_ocr_engine(cfg: config.Config):
    if not cfg.features.ocr:
        return None
    if cfg.ocr_engine == "tesseract":
        import shutil

        if not shutil.which("tesseract"):
            _warn("OCR disabled (tesseract binary not found; install 'tesseract')")
            return None
    try:
        from .models.ocr import get_ocr_engine

        return get_ocr_engine(cfg)
    except Exception as e:
        _warn(f"OCR disabled ({e})")
        return None


class SearchEngine:
    def __init__(self, embedder=None, image_embedder=None, cfg: config.Config | None = None):
        cfg = cfg or config.Config()
        self.embedder = embedder
        self.image_embedder = image_embedder
        self.text_min_similarity = cfg.text_min_similarity
        self.image_min_similarity = cfg.image_min_similarity
        self.con = db.connect(
            config.db_path(), load_vec=embedder is not None or image_embedder is not None
        )

    @classmethod
    def from_config(cls, cfg: config.Config | None = None) -> "SearchEngine":
        cfg = cfg or config.load()
        return cls(build_embedder(cfg), build_image_embedder(cfg), cfg)

    @classmethod
    def fts_only(cls) -> "SearchEngine":
        return cls(None, None)

    def search(
        self, query: str, limit: int = 20, path_prefix: str | None = None
    ) -> list[search.Hit]:
        return search.search(
            self.con,
            query,
            limit=limit,
            embedder=self.embedder,
            image_embedder=self.image_embedder,
            path_prefix=path_prefix,
            text_min_similarity=self.text_min_similarity,
            image_min_similarity=self.image_min_similarity,
        )
