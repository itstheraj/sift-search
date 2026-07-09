from __future__ import annotations
import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

APP = "sift"
PROFILES: dict[str, dict[str, bool]] = {
    "light": {"text": True, "semantic": False, "image": False, "asr": False, "ocr": False},
    "medium": {"text": True, "semantic": True, "image": True, "asr": False, "ocr": False},
    "heavy": {"text": True, "semantic": True, "image": True, "asr": True, "ocr": True},
}
DEFAULT_EXCLUDES = [
    "**/.git/**",
    "**/node_modules/**",
    "**/.venv/**",
    "**/__pycache__/**",
    "**/.cache/**",
    "**/*.tmp",
]


def _xdg(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val) if val else default


def config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", Path.home() / ".config") / APP


def data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", Path.home() / ".local" / "share") / APP


def config_path() -> Path:
    return config_dir() / "config.toml"


def db_path() -> Path:
    return data_dir() / "index.db"


def models_dir() -> Path:
    return data_dir() / "models"


@dataclass
class Features:
    text: bool = True
    semantic: bool = False
    image: bool = False
    asr: bool = False
    ocr: bool = False


@dataclass
class Config:
    profile: str = "heavy"
    folders: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    features: Features = field(default_factory=Features)
    max_workers: int = 4
    nice: int = 10
    ionice_class: int = 3
    device: str = "cpu"
    text_model: str = "BAAI/bge-m3"
    image_model: str = "ViT-B-16-SigLIP2-256"
    image_pretrained: str = "webli"
    asr_model: str = "small"
    ocr_engine: str = "tesseract"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    text_min_similarity: float = 0.0
    image_min_similarity: float = 0.0

    @property
    def folder_paths(self) -> list[Path]:
        return [Path(p).expanduser() for p in self.folders]


def _resolve_features(profile: str, overrides: dict) -> Features:
    base = dict(PROFILES.get(profile, PROFILES["heavy"]))
    for key in ("text", "semantic", "image", "asr", "ocr"):
        if key in overrides:
            base[key] = bool(overrides[key])
    return Features(**base)


def load(path: Path | None = None) -> Config:
    path = path or config_path()
    raw: dict = {}
    if path.exists():
        with path.open("rb") as f:
            raw = tomllib.load(f)
    cfg = Config()
    cfg.profile = raw.get("profile", cfg.profile)
    cfg.folders = list(raw.get("folders", cfg.folders))
    cfg.excludes = list(raw.get("excludes", cfg.excludes))
    res = raw.get("resources", {})
    cfg.max_workers = int(res.get("max_workers", cfg.max_workers))
    cfg.nice = int(res.get("nice", cfg.nice))
    cfg.ionice_class = int(res.get("ionice_class", cfg.ionice_class))
    cfg.device = res.get("device", cfg.device)
    models = raw.get("models", {})
    cfg.text_model = models.get("text", cfg.text_model)
    cfg.image_model = models.get("image", cfg.image_model)
    cfg.image_pretrained = models.get("image_pretrained", cfg.image_pretrained)
    cfg.asr_model = models.get("asr", cfg.asr_model)
    cfg.ocr_engine = models.get("ocr_engine", cfg.ocr_engine)
    chunk = raw.get("chunking", {})
    cfg.chunk_size = int(chunk.get("size", cfg.chunk_size))
    cfg.chunk_overlap = int(chunk.get("overlap", cfg.chunk_overlap))
    srch = raw.get("search", {})
    cfg.text_min_similarity = float(srch.get("text_min_similarity", cfg.text_min_similarity))
    cfg.image_min_similarity = float(srch.get("image_min_similarity", cfg.image_min_similarity))
    cfg.features = _resolve_features(cfg.profile, raw.get("features", {}))
    return cfg


DEFAULT_CONFIG_TEMPLATE = '# Sift configuration.  Profile sets a baseline; [features] flags override it.\nprofile = "heavy"   # light | medium | heavy\n\n# Folders to index (absolute paths or ~).\nfolders = [\n  # "~/Documents",\n  # "~/Videos",\n]\n\nexcludes = [\n  "**/.git/**", "**/node_modules/**", "**/.venv/**",\n  "**/__pycache__/**", "**/.cache/**", "**/*.tmp",\n]\n\n# profile baseline: light = text only; medium = +semantic +image;\n#                   heavy = +audio/video transcription +OCR.\n# [features]            # uncomment to override the profile one flag at a time\n# text = true\n# semantic = true\n# image = true\n# asr = true\n# ocr = true\n\n[resources]\nmax_workers = 4\nnice = 10               # CPU niceness for indexing (0-19, higher = nicer)\nionice_class = 3        # 3 = idle IO priority\ndevice = "cpu"          # cpu = no hardware dependency (default).\n                        # turn on a GPU with "auto" | "rocm" | "vulkan"\n\n[models]\ntext = "BAAI/bge-m3"\nimage = "ViT-B-16-SigLIP2-256"   # higher quality: "ViT-SO400M-16-SigLIP2-384"\nimage_pretrained = "webli"\nasr = "small"                    # higher quality: "large-v3" / "distil-large-v3"\nocr_engine = "tesseract"         # "tesseract" (light) | "onnx" (RapidOCR, heavier)\n\n[chunking]\nsize = 1000\noverlap = 150\n\n[search]\n# Optional cosine floors for vector candidates. Both off by default: with\n# bge-m3, meaningless matches reach 0.544 and loosely worded relevant ones can\n# sit at 0.545, so no safe cutoff exists. Raise only for a corpus you know.\ntext_min_similarity = 0.0\nimage_min_similarity = 0.0\n'


def write_default_config(path: Path | None = None) -> tuple[Path, bool]:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path, False
    path.write_text(DEFAULT_CONFIG_TEMPLATE)
    return path, True


def _toml_list(items: list[str]) -> str:
    return "[" + ", ".join('"' + str(i).replace('"', '\\"') + '"' for i in items) + "]"


def dump(cfg: Config) -> str:
    f = cfg.features
    lines = [
        f'profile = "{cfg.profile}"',
        f"folders = {_toml_list(cfg.folders)}",
        f"excludes = {_toml_list(cfg.excludes)}",
        "",
        "[features]",
        f"text = {str(f.text).lower()}",
        f"semantic = {str(f.semantic).lower()}",
        f"image = {str(f.image).lower()}",
        f"asr = {str(f.asr).lower()}",
        f"ocr = {str(f.ocr).lower()}",
        "",
        "[resources]",
        f"max_workers = {cfg.max_workers}",
        f"nice = {cfg.nice}",
        f"ionice_class = {cfg.ionice_class}",
        f'device = "{cfg.device}"',
        "",
        "[models]",
        f'text = "{cfg.text_model}"',
        f'image = "{cfg.image_model}"',
        f'image_pretrained = "{cfg.image_pretrained}"',
        f'asr = "{cfg.asr_model}"',
        f'ocr_engine = "{cfg.ocr_engine}"',
        "",
        "[chunking]",
        f"size = {cfg.chunk_size}",
        f"overlap = {cfg.chunk_overlap}",
        "",
        "[search]",
        f"text_min_similarity = {cfg.text_min_similarity}",
        f"image_min_similarity = {cfg.image_min_similarity}",
        "",
    ]
    return "\n".join(lines)


def save(cfg: Config, path: Path | None = None) -> Path:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump(cfg))
    return path


def with_profile(cfg: Config, profile: str) -> Config:
    return replace(cfg, profile=profile, features=_resolve_features(profile, {}))
