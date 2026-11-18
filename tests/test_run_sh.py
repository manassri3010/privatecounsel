"""Run benchmarks/run.sh against fake docker, nvidia-smi and curl, so CI can check it without a GPU."""

import json
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "run.sh"

FAKE_NVIDIA_SMI = """#!/usr/bin/env bash
case "$*" in
  *driver_version*) echo "$FAKE_DRIVER" ;;
  *query-gpu=name*) echo "NVIDIA L40S" ;;
  *query-gpu=memory.total*) echo "46068" ;;
  *-l\\ 1*)
    echo "timestamp, memory.used [MiB], utilization.gpu [%], power.draw [W]"
    echo "2026/11/18 10:00:00.000, 41000, 95, 300.1"
    echo "2026/11/18 10:00:01.000, 42500, 97, 310.4"
    exec sleep 30 ;;
  *) echo "fake nvidia-smi $*" ;;
esac
"""

# Records every call; `docker exec ... vllm bench serve` writes a result file into the /results mount.
FAKE_DOCKER = """#!/usr/bin/env bash
echo "docker $*" >> "$FAKE_LOG"
case "$1" in
  run)
    for arg in "$@"; do
      case "$arg" in *:/results) echo "${arg%:/results}" > "$FAKE_STATE" ;; esac
    done ;;
  inspect) echo true ;;
  logs) echo "fake vllm log" ;;
  exec)
    if [[ "$*" == *"import vllm"* ]]; then echo "0.32.0"; exit 0; fi
    results=$(cat "$FAKE_STATE")
    prev=""
    for arg in "$@"; do
      [ "$prev" = "--result-filename" ] && echo '{"median_ttft_ms": 50.0}' > "$results/$arg"
      prev=$arg
    done
    echo "Median TTFT (ms): 50.0" ;;
esac
exit 0
"""


@pytest.fixture
def fake_env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in {"nvidia-smi": FAKE_NVIDIA_SMI, "docker": FAKE_DOCKER, "curl": "#!/bin/sh\nexit 0\n"}.items():
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_LOG": str(tmp_path / "docker.log"),
        "FAKE_STATE": str(tmp_path / "state"),
        "FAKE_DRIVER": "580.65.06",
        "RESULTS_ROOT": str(tmp_path / "results"),
        "HF_CACHE": str(tmp_path / "hf"),
        "CONCURRENCIES": "1 8",
    }


def run(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60)


def test_full_run_writes_results_and_metadata(fake_env, tmp_path):
    r = run(fake_env, "L40S", "Qwen/Qwen3-8B-FP8")
    assert r.returncode == 0, r.stderr

    [run_dir] = (tmp_path / "results").iterdir()
    assert run_dir.name.startswith("L40S_Qwen_Qwen3-8B-FP8_")
    for name in ["nvidia-smi.txt", "gpu_memory_after_load.csv", "gpu_monitor.csv", "vllm.log", "c1.json", "c8.json"]:
        assert (run_dir / name).exists(), name

    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["gpu"] == "L40S"
    assert meta["gpu_model"] == "NVIDIA L40S"
    assert meta["model"] == "Qwen/Qwen3-8B-FP8"
    assert meta["vllm_image"] == "vllm/vllm-openai:v0.32.0"
    assert meta["vllm_version"] == "0.32.0"
    assert meta["gpu_memory_peak_mib"] == 42500
    assert meta["concurrencies"] == [1, 8]


def test_starts_pinned_vllm_with_model_and_extra_args(fake_env):
    run(fake_env, "L4", "Qwen/Qwen3-8B-AWQ", "--gpu-memory-utilization=0.85")
    calls = Path(fake_env["FAKE_LOG"]).read_text()
    [start] = [line for line in calls.splitlines() if line.startswith("docker run -d")]
    assert "vllm/vllm-openai:v0.32.0" in start
    assert "--model=Qwen/Qwen3-8B-AWQ" in start
    assert start.endswith("--gpu-memory-utilization=0.85")
    benches = [line for line in calls.splitlines() if "vllm bench serve" in line]
    assert len(benches) == 2
    assert "--max-concurrency 8" in benches[1]
    assert "--num-prompts 80" in benches[1]  # 10 x concurrency
    assert "--num-prompts 32" in benches[0]  # minimum
    assert "rm -f privatecounsel-bench-vllm" in calls.splitlines()[-1]  # container cleaned up


def test_old_driver_stops_before_pulling_and_suggests_cu129(fake_env):
    r = run({**fake_env, "FAKE_DRIVER": "570.133.20"}, "L4", "Qwen/Qwen3-8B")
    assert r.returncode != 0
    assert "VLLM_TAG=v0.32.0-cu129" in r.stderr
    assert not Path(fake_env["FAKE_LOG"]).exists()


def test_requires_gpu_and_model_arguments(fake_env):
    r = run(fake_env, "L40S")
    assert r.returncode != 0
    assert "usage:" in r.stderr
