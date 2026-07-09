from __future__ import annotations
import re
from typing import Protocol, runtime_checkable
from ..config import Config

PDF_OCR_MIN_CHARS = 64
PDF_RENDER_DPI = 200

OCR_MIN_WORDS = 2
OCR_MIN_WORD_CHARS = 8
OCR_MIN_WORD_RATIO = 0.6
_OCR_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)


def looks_like_text(text: str) -> bool:
    compact = "".join(text.split())
    if not compact:
        return False
    words = _OCR_WORD.findall(text)
    if len(words) < OCR_MIN_WORDS:
        return False
    in_words = sum(len(w) for w in words)
    if in_words < OCR_MIN_WORD_CHARS:
        return False
    return in_words / len(compact) >= OCR_MIN_WORD_RATIO


@runtime_checkable
class OcrEngine(Protocol):
    def ocr_image(self, path: str) -> str: ...

    def ocr_pdf(self, path: str) -> list[dict]: ...


class _BaseOcr:
    def _text(self, image) -> str:
        raise NotImplementedError

    def ocr_image(self, path: str) -> str:
        from PIL import Image

        try:
            with Image.open(path) as img:
                text = self._text(img.convert("RGB")).strip()
        except Exception:
            return ""
        return text if looks_like_text(text) else ""

    def ocr_pdf(self, path: str) -> list[dict]:
        import fitz
        from PIL import Image

        out: list[dict] = []
        zoom = PDF_RENDER_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        with fitz.open(path) as doc:
            for page_no, page in enumerate(doc, start=1):
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                text = self._text(img).strip()
                if looks_like_text(text):
                    out.append({"page": page_no, "text": text})
        return out


class TesseractOcr(_BaseOcr):
    def __init__(self, lang: str = "eng"):
        self.lang = lang
        import pytesseract

        self._pt = pytesseract

    def _text(self, image) -> str:
        return self._pt.image_to_string(image, lang=self.lang)


class OnnxOcr(_BaseOcr):
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def _text(self, image) -> str:
        import numpy as np

        arr = np.array(image)
        result, _ = self._engine(arr)
        if not result:
            return ""
        return "\n".join((line[1] for line in result))


def get_ocr_engine(cfg: Config) -> OcrEngine:
    engine = (cfg.ocr_engine or "tesseract").lower()
    if engine == "onnx":
        return OnnxOcr()
    return TesseractOcr()
