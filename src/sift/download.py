from __future__ import annotations

from .config import Config


def warm_text(cfg: Config) -> None:
    from .models.text_embed import get_embedder

    get_embedder(cfg).encode(["warmup"])


def warm_image(cfg: Config) -> None:
    from .models.image_embed import get_image_embedder

    get_image_embedder(cfg).encode_text(["warmup"])


def warm_asr(cfg: Config) -> None:
    from .models.asr import get_transcriber

    get_transcriber(cfg).ensure_loaded()


def warm_ocr(cfg: Config) -> None:
    from .models.ocr import get_ocr_engine

    get_ocr_engine(cfg)


WARMERS = {"semantic": warm_text, "image": warm_image, "asr": warm_asr, "ocr": warm_ocr}


def download_for(key: str, cfg: Config) -> None:
    WARMERS[key](cfg)


def enabled_downloads(cfg: Config) -> list[tuple[str, str]]:
    f = cfg.features
    out: list[tuple[str, str]] = []
    if f.semantic:
        out.append(("semantic", cfg.text_model))
    if f.image:
        out.append(("image", cfg.image_model))
    if f.asr:
        out.append(("asr", f"faster-whisper-{cfg.asr_model}"))
    if f.ocr and cfg.ocr_engine == "onnx":
        out.append(("ocr", "RapidOCR"))
    return out


def download_enabled(cfg: Config, log=print) -> None:
    items = enabled_downloads(cfg)
    if not items:
        log("No model-backed features enabled; nothing to download.")
        return
    for key, name in items:
        log(f"Downloading {key} model: {name} ...")
        try:
            WARMERS[key](cfg)
            log("  done")
        except Exception as e:  # noqa: BLE001
            log(f"  failed: {e}")
