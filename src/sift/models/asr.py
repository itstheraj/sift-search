from __future__ import annotations
import os
import subprocess
import tempfile
from typing import Protocol, runtime_checkable
from ..config import Config, models_dir


@runtime_checkable
class Transcriber(Protocol):
    def transcribe(self, path: str) -> list[dict]: ...


def _asr_device(pref: str) -> str:
    if pref in ("cpu", "vulkan", "rocm"):
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def extract_wav(src: str, dst: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            src,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            dst,
        ],
        check=True,
        capture_output=True,
    )


class FasterWhisperTranscriber:
    def __init__(self, model_name: str = "small", device: str = "cpu", beam_size: int = 1):
        self.model_name = model_name
        self._device_pref = device
        self.beam_size = beam_size
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            from .pins import ASR_REVISIONS

            dev = _asr_device(self._device_pref)
            compute = "int8" if dev == "cpu" else "float16"
            revision = ASR_REVISIONS.get(self.model_name)
            self._model = WhisperModel(
                self.model_name,
                device=dev,
                compute_type=compute,
                download_root=str(models_dir()),
                **({"revision": revision} if revision else {}),
            )
        return self._model

    @property
    def device(self) -> str:
        return _asr_device(self._device_pref)

    def ensure_loaded(self):
        self._load()

    def transcribe(self, path: str) -> list[dict]:
        model = self._load()
        fd, wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            extract_wav(path, wav)
            segments, _info = model.transcribe(wav, beam_size=self.beam_size)
            out: list[dict] = []
            for s in segments:
                text = (s.text or "").strip()
                if text:
                    out.append(
                        {"text": text, "start_ms": int(s.start * 1000), "end_ms": int(s.end * 1000)}
                    )
            return out
        finally:
            try:
                os.unlink(wav)
            except OSError:
                pass


def get_transcriber(cfg: Config) -> Transcriber:
    return FasterWhisperTranscriber(cfg.asr_model, device=cfg.device)
