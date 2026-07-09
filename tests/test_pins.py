import re

import pytest

from sift.config import Config
from sift.models import pins
from sift.models.asr import FasterWhisperTranscriber
from sift.models.text_embed import SentenceTransformerEmbedder

_SHA = re.compile(r"^[0-9a-f]{40}$")


def test_pins_are_full_commit_shas():
    revisions = [
        *pins.TEXT_REVISIONS.values(),
        *pins.IMAGE_REVISIONS.values(),
        *pins.ASR_REVISIONS.values(),
    ]
    assert revisions
    for rev in revisions:
        assert _SHA.match(rev), rev


def test_config_defaults_are_pinned():
    """A default install must never fetch an unpinned revision."""
    cfg = Config()
    assert cfg.text_model in pins.TEXT_REVISIONS
    assert (cfg.image_model, cfg.image_pretrained) in pins.IMAGE_REVISIONS
    assert cfg.asr_model in pins.ASR_REVISIONS


def test_text_embedder_passes_revision(monkeypatch):
    captured = {}

    class FakeST:
        def __init__(self, name, **kwargs):
            captured.update(kwargs, name=name)

        def get_sentence_embedding_dimension(self):
            return 3

    st = pytest.importorskip("sentence_transformers")
    monkeypatch.setattr(st, "SentenceTransformer", FakeST)

    SentenceTransformerEmbedder("BAAI/bge-m3", device="cpu")._load()
    assert captured["revision"] == pins.TEXT_REVISIONS["BAAI/bge-m3"]


def test_text_embedder_omits_revision_for_unpinned_model(monkeypatch):
    captured = {}

    class FakeST:
        def __init__(self, name, **kwargs):
            captured.update(kwargs, name=name)

        def get_sentence_embedding_dimension(self):
            return 3

    st = pytest.importorskip("sentence_transformers")
    monkeypatch.setattr(st, "SentenceTransformer", FakeST)

    SentenceTransformerEmbedder("some/custom-model", device="cpu")._load()
    assert "revision" not in captured


def test_image_embedder_resolves_pinned_checkpoint(monkeypatch):
    """open_clip's factory has no `revision`, so the checkpoint is resolved by hand."""
    pytest.importorskip("open_clip")
    import open_clip

    from sift.models import image_embed

    captured = {}

    def fake_download(repo_id, filename=None, revision=None, cache_dir=None):
        captured.update(repo_id=repo_id, revision=revision)
        return "/tmp/pinned-checkpoint.bin"

    def fake_create(name, **kwargs):
        captured["pretrained"] = kwargs.get("pretrained")
        captured["image_mean"] = kwargs.get("image_mean")
        captured["image_resize_mode"] = kwargs.get("image_resize_mode")
        return object(), None, object()

    monkeypatch.setattr(open_clip.pretrained, "download_pretrained_from_hf", fake_download)
    monkeypatch.setattr(open_clip, "create_model_and_transforms", fake_create)

    image_embed.OpenClipEmbedder("ViT-B-16-SigLIP2-256", "webli", device="cpu")._create()

    assert captured["repo_id"] == "timm/ViT-B-16-SigLIP2-256"
    assert captured["revision"] == pins.IMAGE_REVISIONS[("ViT-B-16-SigLIP2-256", "webli")]
    assert captured["pretrained"] == "/tmp/pinned-checkpoint.bin"
    # The tag's preprocessing must be forwarded, since passing a path skips it.
    assert captured["image_mean"] == (0.5, 0.5, 0.5)
    assert captured["image_resize_mode"] == "squash"


def test_image_embedder_falls_back_for_unpinned_model(monkeypatch):
    pytest.importorskip("open_clip")
    import open_clip

    from sift.models import image_embed

    captured = {}

    def fake_create(name, **kwargs):
        captured["pretrained"] = kwargs.get("pretrained")
        return object(), None, object()

    monkeypatch.setattr(open_clip, "create_model_and_transforms", fake_create)
    image_embed.OpenClipEmbedder("ViT-B-32", "laion2b_s34b_b79k", device="cpu")._create()
    # Unpinned: hand the tag straight to open_clip, as before.
    assert captured["pretrained"] == "laion2b_s34b_b79k"


def test_asr_passes_revision(monkeypatch):
    captured = {}

    class FakeWhisper:
        def __init__(self, name, **kwargs):
            captured.update(kwargs, name=name)

    fw = pytest.importorskip("faster_whisper")
    monkeypatch.setattr(fw, "WhisperModel", FakeWhisper)

    FasterWhisperTranscriber("small", device="cpu")._load()
    assert captured["revision"] == pins.ASR_REVISIONS["small"]
