from __future__ import annotations
import os
import resource
import tempfile
import threading
import time
from pathlib import Path
from . import config, db, indexer

LOREM = "the quick brown fox jumps over the lazy dog while quarterly budgets and fiscal reports describe revenue across every region of the company ".split()


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(pct / 100 * (len(s) - 1)))))
    return s[k]


def _gen_corpus(root: Path, n_docs: int, words: int) -> int:
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(n_docs):
        body = " ".join((LOREM[(i + j) % len(LOREM)] for j in range(words)))
        text = f"Document {i}\nunique_token_{i} {body}\n"
        p = root / f"doc_{i:05d}.txt"
        p.write_text(text)
        total += len(text)
    return total


def _fs_probe(stop: threading.Event, probe_dir: Path, payload: bytes) -> list[float]:
    probe_dir.mkdir(parents=True, exist_ok=True)
    latencies: list[float] = []
    i = 0
    while not stop.is_set():
        f = probe_dir / f"probe_{i % 8}.bin"
        t0 = time.perf_counter()
        fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 420)
        os.write(fd, payload)
        os.fsync(fd)
        os.close(fd)
        os.stat(f)
        os.unlink(f)
        latencies.append((time.perf_counter() - t0) * 1000)
        i += 1
        time.sleep(0.002)
    return latencies


def _probe_for(duration: float, probe_dir: Path, payload: bytes) -> list[float]:
    stop = threading.Event()
    result: list[list[float]] = []
    t = threading.Thread(target=lambda: result.append(_fs_probe(stop, probe_dir, payload)))
    t.start()
    time.sleep(duration)
    stop.set()
    t.join()
    return result[0] if result else []


def run_bench(n_docs: int = 800, words: int = 400, queries: int = 50) -> int:
    with tempfile.TemporaryDirectory(prefix="sift-bench-") as d:
        base = Path(d)
        corpus = base / "corpus"
        probe_dir = base / "probe"
        payload = os.urandom(64 * 1024)
        print(f"Generating corpus: {n_docs} docs × ~{words} words …")
        total_bytes = _gen_corpus(corpus, n_docs, words)
        cfg = config.with_profile(config.Config(), "light")
        con = db.connect(base / "index.db")
        print("Measuring idle baseline file I/O …")
        idle = _probe_for(2.0, probe_dir, payload)
        print("Reindexing while probing file I/O under load …")
        stop = threading.Event()
        under_box: list[list[float]] = []
        probe_t = threading.Thread(
            target=lambda: under_box.append(_fs_probe(stop, probe_dir, payload))
        )
        probe_t.start()
        t0 = time.perf_counter()
        res = indexer.reindex(con, cfg, [corpus])
        elapsed = time.perf_counter() - t0
        stop.set()
        probe_t.join()
        under = under_box[0] if under_box else []
        from . import search

        qlat: list[float] = []
        for i in range(queries):
            q = f"unique_token_{i % n_docs}"
            tq = time.perf_counter()
            search.search(con, q, limit=20)
            qlat.append((time.perf_counter() - tq) * 1000)
        maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        docs_s = res.indexed / elapsed if elapsed else 0
        mb_s = total_bytes / 1000000.0 / elapsed if elapsed else 0
        p50i, p95i = (_percentile(idle, 50), _percentile(idle, 95))
        p50u, p95u = (_percentile(under, 50), _percentile(under, 95))
        ratio = p95u / p95i if p95i else float("nan")
        print("\n" + "=" * 58)
        print("  SIFT BENCHMARK (light profile)")
        print("=" * 58)
        print(f"  Indexed         : {res.indexed} docs / {total_bytes / 1000000.0:.1f} MB")
        print(f"  Reindex time    : {elapsed:.2f} s")
        print(f"  Throughput      : {docs_s:.0f} docs/s | {mb_s:.1f} MB/s")
        print(
            f"  Query latency   : p50 {_percentile(qlat, 50):.2f} ms | p95 {_percentile(qlat, 95):.2f} ms ({queries} queries)"
        )
        print(f"  Peak RSS        : {maxrss_mb:.0f} MB")
        print("-" * 58)
        print("  File-op latency   p50 (ms)    p95 (ms)    samples")
        print(f"   idle baseline     {p50i:7.3f}     {p95i:7.3f}      {len(idle)}")
        print(f"   under reindex     {p50u:7.3f}     {p95u:7.3f}      {len(under)}")
        print(f"  p95 slowdown      : {ratio:.2f}× (target < 1.5×)")
        print("=" * 58)
        verdict = "PASS" if ratio < 1.5 or p95u < 5.0 else "REVIEW"
        print(f"  Desktop-impact verdict: {verdict}")
        print("  (heavier profiles add model CPU load; this measures the indexer)")
        return 0
