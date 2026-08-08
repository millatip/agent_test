"""Analyze a probe JSONL log produced by attacker_probe/probe.py.

Computes per-probe_type TTFT summary stats, runs Welch's t-tests comparing
known_exact and near_miss_control against the cold_baseline floor, reports
an SNR metric, and saves two plots (distribution comparison + time series).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

PROBE_ORDER = ["cold_baseline", "near_miss_control", "known_exact"]
PROBE_COLORS = {
    "cold_baseline": "#888888",
    "near_miss_control": "#d98c2b",
    "known_exact": "#2b7fd9",
}


def load(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("probe_type")["ttft_ms"]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .reindex(PROBE_ORDER)
    )


def welch_vs_baseline(df: pd.DataFrame, group: str, baseline: str = "cold_baseline"):
    g = df.loc[df["probe_type"] == group, "ttft_ms"]
    b = df.loc[df["probe_type"] == baseline, "ttft_ms"]
    if len(g) < 2 or len(b) < 2:
        return None
    t_stat, p_value = stats.ttest_ind(b, g, equal_var=False)
    snr = (b.mean() - g.mean()) / b.std() if b.std() > 0 else float("nan")
    return {
        "group": group,
        "baseline": baseline,
        "group_mean_ms": g.mean(),
        "baseline_mean_ms": b.mean(),
        "delta_ms": b.mean() - g.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "snr": snr,
        "n_group": len(g),
        "n_baseline": len(b),
    }


def print_report(summary: pd.DataFrame, tests: dict, alpha: float) -> None:
    print("\n=== Per-probe_type TTFT summary (ms) ===")
    print(summary.to_string(float_format=lambda x: f"{x:.1f}"))

    print(f"\n=== Welch's t-test vs. cold_baseline (alpha={alpha}) ===")
    for name, result in tests.items():
        if result is None:
            print(f"{name}: not enough samples to test")
            continue
        sig = result["p_value"] < alpha
        print(
            f"{name}: mean={result['group_mean_ms']:.1f}ms vs "
            f"baseline={result['baseline_mean_ms']:.1f}ms "
            f"(Δ={result['delta_ms']:+.1f}ms, t={result['t_stat']:.2f}, "
            f"p={result['p_value']:.4g}, SNR={result['snr']:.2f}, "
            f"n={result['n_group']}/{result['n_baseline']}) "
            f"-> {'SIGNIFICANT reduction' if sig and result['delta_ms'] > 0 else 'not significant'}"
        )

    known = tests.get("known_exact")
    near = tests.get("near_miss_control")
    print("\n=== Core claim ===")
    if known is not None:
        known_hit = known["p_value"] < alpha and known["delta_ms"] > 0
        print(
            f"known_exact shows a {'statistically significant' if known_hit else 'NOT statistically significant'} "
            f"TTFT reduction vs. cold_baseline (p={known['p_value']:.4g}, SNR={known['snr']:.2f})."
        )
    if near is not None:
        near_hit = near["p_value"] < alpha and near["delta_ms"] > 0
        print(
            f"near_miss_control shows {'a' if near_hit else 'no'} statistically significant TTFT reduction "
            f"vs. cold_baseline (p={near['p_value']:.4g}, SNR={near['snr']:.2f}) — "
            f"{'this UNDERMINES exact-match sensitivity' if near_hit else 'consistent with exact-match-sensitive prefix caching (expected)'}."
        )


def plot_distributions(df: pd.DataFrame, out_path: str) -> None:
    groups = [df.loc[df["probe_type"] == pt, "ttft_ms"].values for pt in PROBE_ORDER]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(groups, patch_artist=True, showmeans=True)
    ax.set_xticks(range(1, len(PROBE_ORDER) + 1))
    ax.set_xticklabels(PROBE_ORDER)
    for patch, pt in zip(bp["boxes"], PROBE_ORDER):
        patch.set_facecolor(PROBE_COLORS[pt])
        patch.set_alpha(0.6)
    for pt, g in zip(PROBE_ORDER, groups):
        x = [list(PROBE_ORDER).index(pt) + 1 + (0.08 * ((i % 5) - 2)) for i in range(len(g))]
        ax.scatter(x, g, color=PROBE_COLORS[pt], alpha=0.5, s=14, zorder=3)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT distribution by probe type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_timeseries(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for pt in PROBE_ORDER:
        sub = df[df["probe_type"] == pt]
        ax.scatter(
            sub["elapsed_run_seconds"], sub["ttft_ms"],
            label=pt, color=PROBE_COLORS[pt], alpha=0.7, s=18,
        )
    ax.set_xlabel("Elapsed probe-run time (s)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT over time by probe type (concurrent victim traffic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/probes.jsonl")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load(args.input)

    summary = summarize(df)
    tests = {
        "known_exact": welch_vs_baseline(df, "known_exact"),
        "near_miss_control": welch_vs_baseline(df, "near_miss_control"),
    }
    print_report(summary, tests, args.alpha)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dist_path = str(out_dir / "ttft_distribution.png")
    ts_path = str(out_dir / "ttft_timeseries.png")
    plot_distributions(df, dist_path)
    plot_timeseries(df, ts_path)

    print("\n=== Plots ===")
    print(f"Distribution plot: {dist_path}")
    print(f"Time-series plot:  {ts_path}")


if __name__ == "__main__":
    main()
