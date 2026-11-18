"""Turn benchmarks/results/ into markdown tables with latency, throughput and cost.

Usage:
    python benchmarks/report.py --price L40S=102 --price L4=49 --inr-per-usd 88.5 > benchmarks/results.md

Prices are INR per GPU-hour excluding GST, keyed by the <gpu-name> given to run.sh.
One table per GPU and model; if the same pair was run more than once, the newest run is used.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

GST_RATE = 0.18
SECONDS_PER_HOUR = 3600


def cost_per_million_tokens(hourly_price: float, tokens_per_second: float) -> float:
    """Price of generating one million tokens when the GPU produces `tokens_per_second` in total.

    A rented GPU costs the same per hour however many users share it, so higher throughput
    (more concurrent users) means a lower cost per token.
    """
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    return hourly_price / (tokens_per_second * SECONDS_PER_HOUR) * 1_000_000


def with_gst(amount: float, rate: float = GST_RATE) -> float:
    return amount * (1 + rate)


@dataclass
class Run:
    path: Path
    metadata: dict
    results: dict[int, dict] = field(default_factory=dict)  # concurrency -> vllm bench serve JSON

    @property
    def key(self) -> tuple[str, str]:
        return self.metadata["gpu"], self.metadata["model"]


def load_runs(results_root: Path) -> list[Run]:
    """Read every finished run folder (one with metadata.json) under results_root."""
    runs = []
    for meta_path in sorted(results_root.glob("*/metadata.json")):
        run = Run(meta_path.parent, json.loads(meta_path.read_text()))
        for result_path in run.path.glob("c*.json"):
            result = json.loads(result_path.read_text())
            run.results[int(result["max_concurrency"])] = result
        if run.results:
            runs.append(run)
    return runs


def latest_per_gpu_and_model(runs: list[Run]) -> tuple[list[Run], int]:
    """Keep the newest run for each (GPU, model). Returns the kept runs and how many were skipped."""
    latest: dict[tuple[str, str], Run] = {}
    for run in sorted(runs, key=lambda r: r.metadata["started"]):
        latest[run.key] = run
    kept = sorted(latest.values(), key=lambda r: r.key)
    return kept, len(runs) - len(kept)


def parse_prices(pairs: list[str]) -> dict[str, float]:
    prices = {}
    for pair in pairs:
        gpu, sep, price = pair.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"--price must look like L40S=102, got {pair!r}")
        prices[gpu] = float(price)
    return prices


def _money(value: float | None, symbol: str, decimals: int) -> str:
    return "n/a" if value is None else f"{symbol}{value:.{decimals}f}"


def render_run(run: Run, hourly_inr: float | None, inr_per_usd: float | None) -> str:
    m = run.metadata
    price_note = f"₹{hourly_inr:g}/GPU-hour excl. GST" if hourly_inr is not None else "no price given"
    lines = [
        f"## {m['gpu']} · {m['model']}",
        "",
        f"{m['input_len']} input / {m['output_len']} output tokens per request · vLLM {m['vllm_version']} · "
        f"driver {m['driver_version']} · peak GPU memory {m['gpu_memory_peak_mib']:,} / "
        f"{m['gpu_memory_total_mib']:,} MiB · {price_note} · run {m['started']}",
        "",
        "| Users | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | ITL p95 (ms) | Tokens/s per user "
        "| Output tokens/s (total) | ₹ / M tokens | ₹ / M incl. GST | $ / M tokens | $ / M incl. GST |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    warnings = []
    for users in sorted(run.results):
        r = run.results[users]
        throughput = r["output_throughput"]
        inr = cost_per_million_tokens(hourly_inr, throughput) if hourly_inr is not None else None
        usd = inr / inr_per_usd if inr is not None and inr_per_usd else None
        lines.append(
            f"| {users} | {r['median_ttft_ms']:.0f} | {r['p95_ttft_ms']:.0f} "
            f"| {r['median_itl_ms']:.1f} | {r['p95_itl_ms']:.1f} "
            f"| {1000 / r['median_tpot_ms']:.1f} | {throughput:.0f} "
            f"| {_money(inr, '₹', 2)} | {_money(with_gst(inr) if inr is not None else None, '₹', 2)} "
            f"| {_money(usd, '$', 3)} | {_money(with_gst(usd) if usd is not None else None, '$', 3)} |"
        )
        if r.get("failed"):
            warnings.append(f"> ⚠️ {r['failed']} failed requests at {users} users; see {run.path.name}/c{users}.log")
    return "\n".join(lines + ([""] + warnings if warnings else []))


def render_report(
    runs: list[Run], prices: dict[str, float], inr_per_usd: float | None, skipped: int = 0
) -> str:
    parts = ["# Benchmark results", ""]
    parts.append(
        "TTFT: time to first token. ITL: gap between generated tokens. Tokens/s per user: 1000 / median "
        "time per output token. Cost: GPU-hour price / (total output tokens/s × 3600) × 1,000,000. "
        f"GST is {GST_RATE:.0%}."
        + (f" USD at ₹{inr_per_usd:g} = $1." if inr_per_usd else "")
    )
    if skipped:
        parts.append(f"\n{skipped} older run(s) of the same GPU and model not shown.")
    for run in runs:
        parts += ["", render_run(run, prices.get(run.metadata["gpu"]), inr_per_usd)]
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--price", action="append", default=[], metavar="GPU=INR",
                        help="INR per GPU-hour excluding GST, e.g. L40S=102 (repeatable)")
    parser.add_argument("--inr-per-usd", type=float, help="exchange rate for the USD columns, e.g. 88.5")
    args = parser.parse_args(argv)

    runs, skipped = latest_per_gpu_and_model(load_runs(args.results))
    if not runs:
        print(f"No finished runs in {args.results}", file=sys.stderr)
        return 1
    print(render_report(runs, parse_prices(args.price), args.inr_per_usd, skipped), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
