from sift.extractors.chunking import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("hello world", size=1000) == ["hello world"]


def test_empty():
    assert chunk_text("   ") == []


def test_splits_with_overlap_on_word_boundary():
    words = " ".join((f"word{i}" for i in range(500)))
    chunks = chunk_text(words, size=100, overlap=20)
    assert len(chunks) > 1
    assert all((len(c) <= 100 for c in chunks))
    assert all((not c.startswith(" ") and (not c.endswith(" ")) for c in chunks))
    joined = " ".join(chunks)
    assert "word0" in joined and "word499" in joined
