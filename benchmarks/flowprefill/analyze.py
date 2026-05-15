# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlowPrefill benchmark analysis.

Consumes the per-trial CSVs + meta.json siblings produced by loadgen.py and
emits the figures + summary.md used in the blog post.

Outputs (all written under <run_dir>/figs/ + <run_dir>/summary.md):

  figs/headline_3line.png      attainment % per (policy, tier) — the headline
  figs/ttft_cdf_urgent.png     CDF of prefill TTFT per policy, urgent tier
  figs/ttft_cdf_generous.png   same, generous tier
  figs/tradeoff_scatter.png    attainment vs throughput across trials
  figs/per_bucket.png          attainment per prompt-length bucket
  figs/preempt_activity.png    preempts/min per policy (parsed from prefill log)
  summary.md                   headline table + paired-difference numbers

Usage:
  python benchmarks/flowprefill/analyze.py <run_dir> \\
      [--log-file /tmp/prefill.log]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_ORDER = ["control", "conservative", "aggressive"]
POLICY_COLORS = {
    "control":      "#666666",
    "conservative": "#1f77b4",
    "aggressive":   "#d62728",
}
TIERS = ["urgent", "generous"]


# vLLM log line format: "INFO 05-15 12:34:56 [file:line] message"
LOG_LINE_RE = re.compile(
    r"^\w+\s+(?P<date>\d{2}-\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+\[[^\]]+\]\s+(?P<msg>.*)$"
)
PREEMPT_INTENT_RE = re.compile(r"PREEMPT INTENT")
RULE2_STUBBORN_RE = re.compile(r"Rule 2 stubborn")


# ──────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────

def discover_trials(run_dir: Path) -> list[dict]:
    """Find every trial_*_policy_*.csv + sibling .meta.json under run_dir."""
    trials = []
    for csv_path in sorted(run_dir.glob("trial_*_policy_*.csv")):
        meta_path = csv_path.with_suffix("").with_suffix(".meta.json")
        if not meta_path.exists():
            # Fallback: filename ends with .csv → strip and add .meta.json
            meta_path = csv_path.parent / (csv_path.stem + ".meta.json")
        if not meta_path.exists():
            print(f"WARN: no meta.json for {csv_path}", file=sys.stderr)
            continue
        with meta_path.open() as f:
            meta = json.load(f)
        df = pd.read_csv(csv_path)
        trials.append({
            "csv_path": csv_path,
            "meta_path": meta_path,
            "meta": meta,
            "df": df,
        })
    return trials


def enrich_dataframes(trials: list[dict]) -> pd.DataFrame:
    """Concat all per-trial DFs into one tall DF with derived columns."""
    frames = []
    for t in trials:
        df = t["df"].copy()
        df["mode"] = t["meta"]["mode"]
        df["trial_id"] = t["meta"]["trial_id"]
        df["rate"] = t["meta"]["rate"]
        # Derived latencies — anchor on server arrival.
        for src, dst in [
            ("t_prefill_done_ms", "prefill_ttft_ms"),
            ("t_decode_first_byte_ms", "e2e_ttft_ms"),
            ("t_decode_last_byte_ms", "total_latency_ms"),
        ]:
            df[dst] = np.where(
                (df[src] > 0) & (df["t_server_arrival_ms"] > 0),
                df[src] - df["t_server_arrival_ms"],
                np.nan,
            )
        df["met_slo"] = df["prefill_ttft_ms"] <= df["slo_ms"]
        df["err_flag"] = df["error"].fillna("").astype(str) != ""
        frames.append(df)
    if not frames:
        raise RuntimeError("No trial dataframes found.")
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────
# Aggregations
# ──────────────────────────────────────────────────────────────────────────

def attainment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(mode, tier, trial) attainment % over measure-window rows."""
    measured = df[(~df["in_warmup"]) & (~df["err_flag"])]
    rows = []
    for (mode, tier, trial_id), grp in measured.groupby(["mode", "tier", "trial_id"]):
        n = len(grp)
        if n == 0:
            continue
        n_met = int(grp["met_slo"].sum())
        rows.append({
            "mode": mode,
            "tier": tier,
            "trial_id": trial_id,
            "n_total": n,
            "n_met": n_met,
            "attainment_pct": 100.0 * n_met / n,
            "throughput_rps": n / (
                (grp["t_client_send_ms"].max() - grp["t_client_send_ms"].min()) / 1000.0
                if n > 1 else 1.0
            ),
        })
    return pd.DataFrame(rows)


def median_with_range(s: pd.Series) -> tuple[float, float, float]:
    return float(s.median()), float(s.min()), float(s.max())


# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────

def plot_headline(att: pd.DataFrame, out: Path) -> None:
    """Three policies × two tiers, median attainment + [min, max] error bars."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(POLICY_ORDER))
    width = 0.35
    for i, tier in enumerate(TIERS):
        meds, lows, highs = [], [], []
        for mode in POLICY_ORDER:
            cell = att[(att["mode"] == mode) & (att["tier"] == tier)]
            if len(cell) == 0:
                meds.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                continue
            m, lo, hi = median_with_range(cell["attainment_pct"])
            meds.append(m); lows.append(m - lo); highs.append(hi - m)
        ax.bar(x + (i - 0.5) * width, meds, width,
               yerr=[lows, highs], capsize=4, label=tier,
               color=("#d62728" if tier == "urgent" else "#2ca02c"))
    ax.set_xticks(x)
    ax.set_xticklabels(POLICY_ORDER)
    ax.set_ylabel("SLO attainment (%)")
    ax.set_title("SLO attainment by policy and tier (median, [min, max] error bars)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_ttft_cdf(df: pd.DataFrame, tier: str, out: Path) -> None:
    measured = df[(df["tier"] == tier) & (~df["in_warmup"]) & (~df["err_flag"])]
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in POLICY_ORDER:
        s = measured[measured["mode"] == mode]["prefill_ttft_ms"].dropna()
        if len(s) == 0:
            continue
        xs = np.sort(s.values)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=f"{mode} (n={len(xs)})",
                color=POLICY_COLORS.get(mode, "black"), linewidth=2)
    ax.set_xlabel("Prefill TTFT (ms)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"TTFT CDF — {tier} tier")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_tradeoff_scatter(att: pd.DataFrame, out: Path) -> None:
    """Attainment % (urgent) vs throughput, one dot per (trial, policy)."""
    urgent = att[att["tier"] == "urgent"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in POLICY_ORDER:
        sub = urgent[urgent["mode"] == mode]
        if len(sub) == 0:
            continue
        ax.scatter(sub["throughput_rps"], sub["attainment_pct"],
                   label=mode, color=POLICY_COLORS.get(mode, "black"),
                   s=80, alpha=0.7)
    ax.set_xlabel("Throughput (req/s, measurement window)")
    ax.set_ylabel("Urgent SLO attainment (%)")
    ax.set_title("Trade-off: urgent attainment vs system throughput")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_per_bucket(df: pd.DataFrame, out: Path) -> None:
    measured = df[(~df["in_warmup"]) & (~df["err_flag"])]
    buckets = sorted(measured["bucket"].dropna().unique())
    if not buckets:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, tier in zip(axes, TIERS):
        sub = measured[measured["tier"] == tier]
        x = np.arange(len(buckets))
        width = 0.25
        for i, mode in enumerate(POLICY_ORDER):
            ms = sub[sub["mode"] == mode]
            ys = []
            for b in buckets:
                cell = ms[ms["bucket"] == b]
                ys.append(100.0 * cell["met_slo"].mean() if len(cell) else np.nan)
            ax.bar(x + (i - 1) * width, ys, width, label=mode,
                   color=POLICY_COLORS.get(mode, "black"))
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in buckets])
        ax.set_title(f"{tier}")
        ax.set_xlabel("prompt-length bucket (tokens)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("attainment (%)")
    axes[0].legend()
    fig.suptitle("Per-bucket SLO attainment")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Log parsing for preempt-activity plot
# ──────────────────────────────────────────────────────────────────────────

def parse_log_events(
    log_path: Path,
    trials: list[dict],
) -> pd.DataFrame:
    """Walk prefill log; bucket each PREEMPT INTENT / Rule 2 stubborn event
    into its enclosing trial by timestamp range.

    Returns a DataFrame with columns: mode, trial_id, ts_ms, event_type.
    """
    if not log_path.exists():
        print(f"[analyze] log file {log_path} not found, skipping preempt plot.",
              file=sys.stderr)
        return pd.DataFrame()

    # Build a list of (start_ms, end_ms, mode, trial_id) windows.
    windows = []
    for t in trials:
        m = t["meta"]
        start = m.get("started_at_ms", 0)
        end = m.get("ended_at_ms", 0)
        if start > 0 and end > 0:
            windows.append((start, end, m["mode"], m["trial_id"]))
    if not windows:
        return pd.DataFrame()

    # Pick a reference year from any trial start. Log lines are MM-DD HH:MM:SS
    # only; we need to splice the year in to convert to epoch.
    any_start_ms = windows[0][0]
    ref_year = datetime.fromtimestamp(any_start_ms / 1000.0, tz=timezone.utc).year

    events = []
    with log_path.open("r", errors="replace") as f:
        for line in f:
            m = LOG_LINE_RE.match(line)
            if not m:
                continue
            is_preempt = bool(PREEMPT_INTENT_RE.search(m["msg"]))
            is_rule2 = bool(RULE2_STUBBORN_RE.search(m["msg"]))
            if not (is_preempt or is_rule2):
                continue
            try:
                dt = datetime.strptime(
                    f"{ref_year}-{m['date']} {m['time']}", "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            ts_ms = int(dt.timestamp() * 1000)
            # Find window. Log may span trial windows; pick the enclosing one.
            for start, end, mode, trial_id in windows:
                if start <= ts_ms <= end:
                    events.append({
                        "mode": mode,
                        "trial_id": trial_id,
                        "ts_ms": ts_ms,
                        "rel_s": (ts_ms - start) / 1000.0,
                        "event_type": "preempt" if is_preempt else "rule2_stubborn",
                    })
                    break
    return pd.DataFrame(events)


def plot_preempt_activity(events: pd.DataFrame, out: Path) -> None:
    if events.empty:
        # Still emit a placeholder so the report is complete.
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "no preempt events found in log",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return

    # Preempts per minute per (mode, trial). Aggregate across trials at the
    # mode level (mean of per-trial counts) for the headline view.
    events = events.copy()
    events["minute"] = (events["rel_s"] // 60).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: preempts/min per mode (averaged across trials).
    ax = axes[0]
    for mode in POLICY_ORDER:
        sub = events[(events["mode"] == mode) & (events["event_type"] == "preempt")]
        if sub.empty:
            continue
        per_trial = sub.groupby(["trial_id", "minute"]).size().reset_index(name="n")
        agg = per_trial.groupby("minute")["n"].mean()
        ax.plot(agg.index, agg.values, marker="o", linewidth=2, label=mode,
                color=POLICY_COLORS.get(mode, "black"))
    ax.set_xlabel("Minute within trial (relative)")
    ax.set_ylabel("PREEMPT INTENT events / min (mean across trials)")
    ax.set_title("Preempt activity over time")
    ax.grid(alpha=0.3)
    ax.legend()

    # Right: ratio of Rule 2 blocks to total monitor intents per mode.
    ax = axes[1]
    summary = []
    for mode in POLICY_ORDER:
        sub = events[events["mode"] == mode]
        n_preempt = (sub["event_type"] == "preempt").sum()
        n_rule2 = (sub["event_type"] == "rule2_stubborn").sum()
        rate = (100.0 * n_rule2 / (n_preempt + n_rule2)) if (n_preempt + n_rule2) else 0.0
        summary.append((mode, n_preempt, n_rule2, rate))
    modes = [s[0] for s in summary]
    rates = [s[3] for s in summary]
    bars = ax.bar(modes, rates,
                  color=[POLICY_COLORS.get(m, "black") for m in modes])
    ax.set_ylabel("Rule-2 stubborn rate (%)")
    ax.set_title("Stubborn-block share of all preempt signals")
    ax.set_ylim(0, max(100, max(rates) * 1.2 if rates else 100))
    ax.grid(axis="y", alpha=0.3)
    for bar, (_, np_, n2, _r) in zip(bars, summary):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"P={np_}\nR2={n2}",
                ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────

def write_summary(
    run_dir: Path,
    df: pd.DataFrame,
    att: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    out = run_dir / "summary.md"
    lines = [
        f"# FlowPrefill benchmark — {run_dir.name}",
        "",
        "## Attainment by policy and tier",
        "",
        "| policy | tier | trials | median % | min % | max % |",
        "|---|---|---|---|---|---|",
    ]
    for mode in POLICY_ORDER:
        for tier in TIERS:
            cell = att[(att["mode"] == mode) & (att["tier"] == tier)]
            if cell.empty:
                continue
            m, lo, hi = median_with_range(cell["attainment_pct"])
            lines.append(
                f"| {mode} | {tier} | {len(cell)} | {m:.1f} | {lo:.1f} | {hi:.1f} |"
            )

    # Paired differences vs control on urgent.
    lines += [
        "",
        "## Paired differences vs control (urgent)",
        "",
        "Per-trial difference: `attainment[policy] - attainment[control]`.",
        "",
        "| policy | trials paired | median Δpp | min Δpp | max Δpp |",
        "|---|---|---|---|---|",
    ]
    ctrl = att[(att["mode"] == "control") & (att["tier"] == "urgent")] \
        .set_index("trial_id")["attainment_pct"]
    for mode in ["conservative", "aggressive"]:
        sub = att[(att["mode"] == mode) & (att["tier"] == "urgent")] \
            .set_index("trial_id")["attainment_pct"]
        common = ctrl.index.intersection(sub.index)
        if len(common) == 0:
            continue
        diffs = sub.loc[common] - ctrl.loc[common]
        lines.append(
            f"| {mode} | {len(common)} | "
            f"{diffs.median():+.1f} | {diffs.min():+.1f} | {diffs.max():+.1f} |"
        )

    if not events.empty:
        lines += ["", "## Preempt activity", "", "| policy | preempts | rule-2 blocks |",
                  "|---|---|---|"]
        for mode in POLICY_ORDER:
            sub = events[events["mode"] == mode]
            np_ = int((sub["event_type"] == "preempt").sum())
            n2 = int((sub["event_type"] == "rule2_stubborn").sum())
            lines.append(f"| {mode} | {np_} | {n2} |")

    lines += [
        "",
        "## Counts",
        "",
        f"- total rows: {len(df)}",
        f"- measure-window rows: {((~df['in_warmup']) & (~df['err_flag'])).sum()}",
        f"- error rows: {df['err_flag'].sum()}",
    ]
    out.write_text("\n".join(lines) + "\n")


# ──────────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Directory containing trial_*.csv + .meta.json")
    ap.add_argument("--log-file", default="/tmp/prefill.log",
                    help="Prefill log used for the preempt-activity plot.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"ERROR: {run_dir} is not a directory", file=sys.stderr)
        return 1

    trials = discover_trials(run_dir)
    if not trials:
        print(f"ERROR: no trials found in {run_dir}", file=sys.stderr)
        return 1
    print(f"[analyze] discovered {len(trials)} trial files in {run_dir}")

    df = enrich_dataframes(trials)
    att = attainment_table(df)

    figs_dir = run_dir / "figs"
    figs_dir.mkdir(exist_ok=True)

    plot_headline(att, figs_dir / "headline_3line.png")
    plot_ttft_cdf(df, "urgent", figs_dir / "ttft_cdf_urgent.png")
    plot_ttft_cdf(df, "generous", figs_dir / "ttft_cdf_generous.png")
    plot_tradeoff_scatter(att, figs_dir / "tradeoff_scatter.png")
    plot_per_bucket(df, figs_dir / "per_bucket.png")

    events = parse_log_events(Path(args.log_file), trials)
    plot_preempt_activity(events, figs_dir / "preempt_activity.png")

    write_summary(run_dir, df, att, events)
    print(f"[analyze] wrote figs to {figs_dir}/ and summary to {run_dir}/summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
