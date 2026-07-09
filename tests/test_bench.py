from sift import bench


def test_percentile():
    assert bench._percentile([], 50) == 0.0
    assert bench._percentile([1, 2, 3, 4], 50) in (2, 3)
    assert bench._percentile([5], 95) == 5


def test_bench_runs_and_reports(capsys):
    rc = bench.run_bench(n_docs=15, words=15, queries=3)
    assert rc == 0
    out = capsys.readouterr().out
    assert "SIFT BENCHMARK" in out
    assert "Throughput" in out
    assert "Desktop-impact verdict" in out
