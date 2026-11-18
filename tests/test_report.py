import json

import pytest
from report import cost_per_million_tokens, latest_per_gpu_and_model, load_runs, main, with_gst


def write_run(root, name, *, gpu="L40S", model="Qwen/Qwen3-8B-FP8", started="2026-12-08T10:00:00Z",
              output_throughput=1000.0):
    run = root / name
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({
        "gpu": gpu, "model": model, "started": started, "input_len": 1024, "output_len": 256,
        "vllm_version": "0.32.0", "driver_version": "580.65.06",
        "gpu_memory_peak_mib": 42500, "gpu_memory_total_mib": 46068,
    }))
    (run / "c8.json").write_text(json.dumps({
        "max_concurrency": 8, "output_throughput": output_throughput, "failed": 0,
        "median_ttft_ms": 120.4, "p95_ttft_ms": 250.6, "median_itl_ms": 12.5, "p95_itl_ms": 20.0,
        "median_tpot_ms": 12.5,
    }))
    return run


def test_cost_per_million_tokens():
    # ₹102/hour at 1000 tokens/s: 3.6M tokens per hour, so ₹102 / 3.6 per million.
    assert cost_per_million_tokens(102, 1000) == pytest.approx(28.3333, abs=1e-4)
    # Doubling throughput halves the cost.
    assert cost_per_million_tokens(102, 2000) == pytest.approx(cost_per_million_tokens(102, 1000) / 2)


def test_cost_rejects_zero_throughput():
    with pytest.raises(ValueError):
        cost_per_million_tokens(102, 0)


def test_with_gst_adds_18_percent():
    assert with_gst(100) == pytest.approx(118)


def test_report_from_fake_results(tmp_path, capsys):
    write_run(tmp_path, "L40S_Qwen_Qwen3-8B-FP8_20261208T100000Z")
    assert main(["--results", str(tmp_path), "--price", "L40S=102", "--inr-per-usd", "85"]) == 0
    out = capsys.readouterr().out
    assert "## L40S · Qwen/Qwen3-8B-FP8" in out
    # users | TTFT p50 | TTFT p95 | ITL p50 | ITL p95 | tok/s per user | total | ₹ | ₹+GST | $ | $+GST
    assert "| 8 | 120 | 251 | 12.5 | 20.0 | 80.0 | 1000 | ₹28.33 | ₹33.43 | $0.333 | $0.393 |" in out


def test_gpu_without_a_price_shows_na(tmp_path, capsys):
    write_run(tmp_path, "run", gpu="L4")
    main(["--results", str(tmp_path), "--price", "L40S=102", "--inr-per-usd", "85"])
    out = capsys.readouterr().out
    assert "| 1000 | n/a | n/a | n/a | n/a |" in out
    assert "no price given" in out


def test_usd_columns_need_exchange_rate(tmp_path, capsys):
    write_run(tmp_path, "run")
    main(["--results", str(tmp_path), "--price", "L40S=102"])
    assert "| ₹28.33 | ₹33.43 | n/a | n/a |" in capsys.readouterr().out


def test_latest_run_per_gpu_and_model_wins(tmp_path):
    write_run(tmp_path, "old", started="2026-12-08T10:00:00Z", output_throughput=500)
    write_run(tmp_path, "new", started="2026-12-09T10:00:00Z", output_throughput=900)
    write_run(tmp_path, "other", gpu="L4")
    kept, skipped = latest_per_gpu_and_model(load_runs(tmp_path))
    assert skipped == 1
    assert [r.path.name for r in kept] == ["other", "new"]


def test_unfinished_run_without_metadata_is_ignored(tmp_path, capsys):
    (tmp_path / "crashed").mkdir()
    assert main(["--results", str(tmp_path)]) == 1
