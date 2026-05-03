"""
Generate SVG charts from benchmark JSON results for the self-inference series.

Reads per-concurrency JSON files from --results-dir and writes SVGs to the
same directory. Run this after benchmarking to produce charts for the post.

Usage:
    uv run python scripts/plot_results.py --results-dir benchmarks/results/post-01

Output (written to --results-dir):
    throughput.svg      — throughput (tok/s) vs concurrency, with theoretical max
    latency.svg         — p50 / p90 / p99 latency vs concurrency
    gpu-utilization.svg — GPU utilization % over time (skipped if no GPU data)
"""

import argparse
import json
from pathlib import Path

import altair as alt

# RTX 4090: 1008 GB/s memory bandwidth ÷ 16 GB (Llama 3.1 8B in bf16)
THEORETICAL_MAX_TOK_S = 63


def load_results(results_dir: Path) -> dict[int, dict]:
    """Load all concurrency-N.json files from the results directory."""
    results = {}
    for path in sorted(results_dir.glob("concurrency-*.json")):
        concurrency = int(path.stem.split("-")[1])
        with open(path) as f:
            data = json.load(f)
        results[concurrency] = data
    return results


def throughput_chart(results: dict) -> alt.Chart:
    """Line + point chart: throughput (tok/s) vs concurrency.
    Failed runs shown in red. Theoretical max shown as a dashed reference line.
    """
    concurrency_levels = sorted(results.keys())
    sort_order = [str(c) for c in concurrency_levels]

    rows = []
    for c in concurrency_levels:
        d = results[c]
        s = d.get("summary", {})
        if "error" in s:
            continue
        rows.append({
            "concurrency": str(c),
            "tok_s": s["throughput_tokens_per_s"],
            "failed": s["num_failed"] > 0,
        })

    x_enc = alt.X("concurrency:O", title="Concurrency",
                   sort=sort_order, axis=alt.Axis(labelAngle=0))

    max_tok_s = max(r["tok_s"] for r in rows) if rows else THEORETICAL_MAX_TOK_S
    y_max = max(max_tok_s * 1.15, THEORETICAL_MAX_TOK_S * 1.15)

    line = alt.Chart(alt.Data(values=rows)).mark_line(
        strokeWidth=2, color="#4C78A8"
    ).encode(
        x=x_enc,
        y=alt.Y("tok_s:Q", title="Throughput (tok/s)", scale=alt.Scale(domain=[0, y_max])),
    )

    points = alt.Chart(alt.Data(values=rows)).mark_point(
        filled=True, size=90
    ).encode(
        x=alt.X("concurrency:O", sort=sort_order),
        y=alt.Y("tok_s:Q"),
        color=alt.condition(
            "datum.failed",
            alt.value("#E45756"),
            alt.value("#4C78A8"),
        ),
        tooltip=[
            alt.Tooltip("concurrency:O", title="Concurrency"),
            alt.Tooltip("tok_s:Q", title="Throughput (tok/s)", format=".1f"),
        ],
    )

    ref_line = alt.Chart(alt.Data(values=[{"y": THEORETICAL_MAX_TOK_S}])).mark_rule(
        strokeDash=[6, 4], color="#aaa", strokeWidth=1
    ).encode(y="y:Q")

    ref_label = alt.Chart(
        alt.Data(values=[{"concurrency": sort_order[-1], "y": THEORETICAL_MAX_TOK_S}])
    ).mark_text(
        align="right", dx=-6, dy=-10, color="#999", fontSize=11,
        text=f"theoretical max (~{THEORETICAL_MAX_TOK_S} tok/s)",
    ).encode(
        x=alt.X("concurrency:O", sort=sort_order),
        y=alt.Y("y:Q"),
    )

    return (line + points + ref_line + ref_label).properties(
        title=alt.TitleParams("Throughput vs Concurrency", anchor="start"),
        width=480, height=260,
    ).configure_view(strokeWidth=0)


def latency_chart(results: dict) -> alt.Chart:
    """Line chart: p50 / p90 / p99 latency vs concurrency.
    Excludes heavily failed runs (>50% failure rate).
    """
    concurrency_levels = sorted(results.keys())
    sort_order = [str(c) for c in concurrency_levels]

    rows = []
    for c in concurrency_levels:
        d = results[c]
        s = d.get("summary", {})
        if "error" in s:
            continue
        if s["num_failed"] > s["num_prompts"] * 0.5:
            continue
        for label, key in [("p50", "latency_p50_s"), ("p90", "latency_p90_s"), ("p99", "latency_p99_s")]:
            rows.append({
                "concurrency": str(c),
                "percentile": label,
                "latency_s": s[key],
            })

    color_scale = alt.Scale(
        domain=["p50", "p90", "p99"],
        range=["#4C78A8", "#F58518", "#E45756"],
    )

    return alt.Chart(alt.Data(values=rows)).mark_line(
        point=True, strokeWidth=2
    ).encode(
        x=alt.X("concurrency:O", title="Concurrency", sort=sort_order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("latency_s:Q", title="Latency (s)"),
        color=alt.Color("percentile:N", scale=color_scale, title="Percentile"),
        tooltip=[
            alt.Tooltip("concurrency:O", title="Concurrency"),
            alt.Tooltip("percentile:N", title="Percentile"),
            alt.Tooltip("latency_s:Q", title="Latency (s)", format=".1f"),
        ],
    ).properties(
        title=alt.TitleParams("Latency vs Concurrency (p50 / p90 / p99)", anchor="start"),
        width=480, height=260,
    ).configure_view(strokeWidth=0)


def gpu_utilization_chart(results: dict) -> alt.Chart | None:
    """Area + line time series: GPU utilization % during a benchmark run.
    Uses the highest non-failed concurrency level available.
    Returns None if no GPU data is present (e.g. CPU-only runs).
    """
    # Pick the highest concurrency level with real GPU data
    target = None
    for c in sorted(results.keys(), reverse=True):
        samples = results[c].get("gpu_samples", [])
        if samples and any(s["utilization_pct"] > 0 for s in samples):
            target = c
            break

    if target is None:
        return None

    samples = results[target]["gpu_samples"]
    rows = [{"t_min": round(s["t"] / 60, 2), "util": s["utilization_pct"]} for s in samples]

    area = alt.Chart(alt.Data(values=rows)).mark_area(
        opacity=0.25, color="#4C78A8"
    ).encode(
        x=alt.X("t_min:Q", title="Time (minutes)"),
        y=alt.Y("util:Q", title="GPU Utilization (%)", scale=alt.Scale(domain=[0, 100])),
    )

    line = alt.Chart(alt.Data(values=rows)).mark_line(
        color="#4C78A8", strokeWidth=2
    ).encode(
        x="t_min:Q",
        y="util:Q",
        tooltip=[
            alt.Tooltip("t_min:Q", title="Time (min)", format=".1f"),
            alt.Tooltip("util:Q", title="GPU Util (%)", format=".0f"),
        ],
    )

    return (area + line).properties(
        title=alt.TitleParams(f"GPU Utilization Over Time (concurrency={target})", anchor="start"),
        width=480, height=220,
    ).configure_view(strokeWidth=0)


def main():
    parser = argparse.ArgumentParser(description="Generate SVG plots from benchmark results")
    parser.add_argument("--results-dir", required=True, help="Directory with concurrency-N.json files")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    if not results:
        print(f"No concurrency-N.json files found in {results_dir}")
        return

    print(f"Loaded results for concurrency levels: {sorted(results.keys())}")

    charts = [
        ("throughput.svg", throughput_chart(results)),
        ("latency.svg", latency_chart(results)),
    ]

    gpu_chart = gpu_utilization_chart(results)
    if gpu_chart:
        charts.append(("gpu-utilization.svg", gpu_chart))
    else:
        print("Skipping GPU chart — no GPU data (CPU-only run?)")

    for filename, chart in charts:
        out_path = results_dir / filename
        chart.save(str(out_path))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
