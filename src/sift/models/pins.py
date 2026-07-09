"""Pinned upstream revisions for the model weights Sift downloads.

Weights are fetched from Hugging Face on first use. Pinning the commit SHA means
a compromised or hijacked upstream repository cannot silently swap the weights
out from under an existing install. A model that is not listed here is fetched
unpinned, which is the price of letting users point `[models]` at anything.

To update a pin, verify the new revision, then replace the SHA:

    curl -s https://huggingface.co/api/models/BAAI/bge-m3 | jq -r .sha
"""

from __future__ import annotations

# sentence-transformers repo id -> commit SHA
TEXT_REVISIONS: dict[str, str] = {
    "BAAI/bge-m3": "5617a9f61b028005a4858fdac845db406aefb181",
}

# (open_clip model name, pretrained tag) -> commit SHA of the backing HF repo
IMAGE_REVISIONS: dict[tuple[str, str], str] = {
    ("ViT-B-16-SigLIP2-256", "webli"): "fb14461786ff5cd7b18210817522546f28c7143a",
    ("ViT-SO400M-16-SigLIP2-384", "webli"): "fec784dabb3081a5f101fc74eefaf9d1ed08237b",
}

# faster-whisper size alias -> commit SHA of the Systran/* repo it resolves to
ASR_REVISIONS: dict[str, str] = {
    "small": "536b0662742c02347bc0e980a01041f333bce120",
    "large-v3": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    "large": "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    "distil-large-v3": "c3058b475261292e64a0412df1d2681c06260fab",
}
