"""Flux-correction adapter of Thesis_model/analysis/report_figures/chapter_6.py.

Plots saved results only. No model training or forecast execution.
python analysis/chapter_6.py
"""
import argparse
import csv
from io import BytesIO
import os
from pathlib import Path
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO = next(p for p in Path(__file__).resolve().parents if (p / "src/Surge.jl").is_file())
LABELS = {"one_step": "127 detached steps", "short": "Short-16", "fixed32": "Fixed-32",
          "fixed64": "Fixed-64", "long": "Long-128", "balanced": "Multi-step",
          "curriculum": "Original curriculum", "detached": "112 detached steps", "frozen": "Frozen"}
COLORS = ["#AA4499", "#D55E00", "#CC79A7", "#0072B2", "#009E73", "#7653A2",
          "#8C6500", "#555555", "#000000"]
MARKERS = ["+", "o", "s", "D", "^", "v", "P", "X", "*"]
STYLES = {name: dict(color=color, marker=marker) for name, color, marker in zip(LABELS, COLORS, MARKERS)}


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def save(fig, output, name):
    for extension in ("png", "pdf", "svg"):
        # Prepare a complete file before replacing the version used by the preview.
        data = BytesIO()
        fig.savefig(data, format=extension, dpi=300, bbox_inches="tight", facecolor="white")
        payload = data.getvalue()
        if extension == "svg":
            payload = b"\n".join(line.rstrip() for line in payload.splitlines()) + b"\n"
        descriptor, temporary = tempfile.mkstemp(dir=output, suffix=f".{extension}")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
            for attempt in range(10):
                try:
                    os.replace(temporary, output / f"{name}.{extension}")
                    break
                except OSError:
                    if attempt==9:
                        raise
                    time.sleep(.2)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    plt.close(fig)
    print(f"Saved {name} (PNG, PDF, SVG)")


def expected_keys():
    return {(fold, repeat, event) for fold in range(1, 6) for repeat in range(1, 4)
            for event in range((fold-1)*18//5+1, fold*18//5+1)}


def average(sample, metric):
    keys = [(int(r["fold"]), int(r["repeat"]), int(r["event"])) for r in sample]
    if len(keys) != 54 or set(keys) != expected_keys():
        raise ValueError("Incomplete or duplicate fold/repeat/event coverage")
    values = [np.mean([float(r[metric]) for r in sample if int(r["event"]) == event])
              for event in range(1, 19)]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite forecast errors; inspect results instead of omitting them")
    return float(np.mean(values))


def summary_value(summary, events, stage, condition, model, start, metric):
    matches = [r for r in summary if int(r["stage"]) == stage and r["condition"] == condition
               and r["model"] == model and float(r["from_hours"]) == start]
    if len(matches) != 1:
        raise ValueError(f"Expected one summary row: {stage}/{condition}/{model}/{start}")
    sample = [r for r in events if int(r["stage"]) == stage and r["condition"] == condition
              and r["model"] == model and float(r["from_hours"]) == start]
    value = average(sample, metric)
    if not np.isclose(value, float(matches[0][metric]), rtol=1e-10, atol=1e-14):
        raise ValueError("Summary disagrees with equal-event aggregation")
    return value


def cost_figure(data, output, name, description):
    fig, ax = plt.subplots(figsize=(8.5, 6.3))
    for objective, row in data.items():
        ax.scatter(row["seconds"], 1000*np.sqrt(row["mse_h"]), **STYLES[objective], s=75, zorder=3)
    ax.set(title=description, xlabel="Mean training-loop time per fit (s)", yscale="log",
           ylabel=r"Water-level RMSE (mm)")
    ax.grid(alpha=.18, which="both")
    handles = [Line2D([], [], **STYLES[n], linestyle="none", markersize=8, label=LABELS[n]) for n in data]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Flux correction: five folds, three repeats; equal weight per event")
    fig.tight_layout(rect=(0, .19, 1, .94))
    save(fig, output, name)


def comparison_figure(output, summary, events, stage, labels, filename):
    """Direct early/late comparison, independent of computation time."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
    names = list(labels)
    colors = ["#999999" if n == "frozen" else COLORS[i % (len(COLORS)-1)]
              for i, n in enumerate(names)]
    for ax, start, title in zip(axes, (0., 48.), ("(a) 0–12 h", "(b) 48–72 h")):
        baseline_condition = "full" if stage == 1 else "long"
        values = [summary_value(summary, events, stage,
                  baseline_condition if n == "frozen" else n,
                  "pretrained" if n == "frozen" else "adapted", start, "mse_h") for n in names]
        values = [1000*np.sqrt(v) for v in values]
        if any(v <= 0 for v in values):
            raise ValueError("Logarithmic comparison requires positive RMSE")
        positions = np.arange(len(names))
        ax.barh(positions, values, color=colors)
        ax.set(yticks=positions, yticklabels=[labels[n] for n in names],
               xlabel=r"Water-level RMSE (mm)", title=title, xscale="log")
        ax.set_xlim(min(values)*.35, max(values)*4)
        for position, value in zip(positions, values):
            ax.text(value*1.12, position, f"{value:.3g}", va="center", fontsize=9)
        ax.grid(axis="x", alpha=.18, which="both")
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Adaptation parameterisations" if stage == 1 else "Temporal objectives")
    fig.text(.5, .02, "Five folds, three repeats; repeats averaged within each of 18 held-out events.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, output, filename)


def local_run(root, job):
    # Saved paths refer to the training device. Reconstruct from scientific keys.
    name = (f'{job["method"]}__{job["temporal"]}__{job["profile"]}__{job["layout"]}'
            f'__noise{round(100*float(job["noise_std"]))}cm')
    return root / "runs" / name / f'fold_{job["fold"]}' / f'repeat_{job["repeat"]}'


def temporal_figures(root, output, summary, events, jobs):
    costs = {}
    for objective in LABELS:
        if objective == "frozen":
            costs[objective] = 0.0
            continue
        fits = [j for j in jobs if int(j["stage"]) == 2 and j["temporal"] == objective]
        if len(fits) != 15 or len({(j["fold"], j["repeat"]) for j in fits}) != 15:
            raise ValueError(f"Expected 15 temporal fits: {objective}")
        costs[objective] = np.mean([float(j["training_seconds"]) for j in fits])
    for start, description, filename in [(0., "0–12 h average", "forecast_length_computation_cost_0to12h"),
                                          (48., "48–72 h average", "forecast_length_computation_cost_48to72h")]:
        data = {n: {"seconds": costs[n], **{
            metric: summary_value(summary, events, 2, "long" if n == "frozen" else n,
                                  "pretrained" if n == "frozen" else "adapted", start, metric)
            for metric in ("mse_h",)}} for n in LABELS}
        cost_figure(data, output, filename, description)
    # Lead 64 is an exact saved timestep, distinct from a 12-hour endpoint.
    data = {}
    for objective in LABELS:
        sample = []
        source = "long" if objective == "frozen" else objective
        label = "pretrained" if objective == "frozen" else "adapted"
        for job in [j for j in jobs if int(j["stage"]) == 2 and j["temporal"] == source]:
            folder = local_run(root, job) / "diagnostics"
            for event in range((int(job["fold"])-1)*18//5+1, int(job["fold"])*18//5+1):
                path = folder / f"{label}_event_{event}.tsv"
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    row = next((r for r in csv.DictReader(stream, delimiter="\t") if int(r["step"]) == 64), None)
                if row is None:
                    raise ValueError(f"Missing lead 64: {path}")
                sample.append(dict(row, fold=job["fold"], repeat=job["repeat"], event=event))
        data[objective] = {"seconds": costs[objective], **{metric: average(sample, metric)
                             for metric in ("mse_h",)}}
    cost_figure(data, output, "forecast_length_computation_cost", "Lead-64")
    with (output / "lead64_accuracy_cost.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["objective", "seconds", "mse_h"], delimiter="\t")
        writer.writeheader()
        writer.writerows(dict(objective=name, **row) for name, row in data.items())


def curriculum_figure(root, output):
    rows = read_tsv(root / "runs/full__curriculum__uniform__both__noise0cm/fold_1/repeat_1/training_history.tsv")
    points = {int(r["epoch"]): float(r["validation_loss"]) for r in rows}
    boundaries = [0, 50, 75, 100, 200]
    values = [1000*np.sqrt(points[e]) for e in boundaries]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite curriculum validation score")
    fig, axes = plt.subplots(2, 1, figsize=(15/2.54, 5.2))
    finite = [r for r in rows if np.isfinite(float(r["validation_loss"]))]
    axes[0].plot([int(r["epoch"]) for r in finite], [1000*np.sqrt(float(r["validation_loss"])) for r in finite], color="#0072B2")
    axes[0].set(title="(a) Sparse validation RMSE", xlabel="Epoch", ylabel=r"Validation RMSE (mm)", yscale="log")
    for boundary in boundaries[1:-1]:
        axes[0].axvline(boundary, color="gray", linestyle=":", alpha=.6)
    reductions = np.array(values[:-1])-np.array(values[1:])
    axes[1].bar(["16\n1–50", "16,32\n51–75", "16,32,64\n76–100", "All\n101–200"], reductions,
                color=["#D55E00" if r < 0 else "#009E73" for r in reductions])
    axes[1].axhline(0, color="#555555", linewidth=.8)
    axes[1].set(title="(b) Change across each curriculum phase", xlabel="Active forecast leads / epoch range",
                ylabel="RMSE reduction (mm)")
    fig.tight_layout()
    save(fig, output, "curriculum_validation_progress")


def noise_figure(output, summary, events):
    levels = [0, 1, 5, 10, 20]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for start, color, marker, label in [(0., "#0072B2", "o", "0–12 h"), (48., "#D55E00", "s", "48–72 h")]:
        means, frozen = [], []
        for level in levels:
            stage, condition = (3, "uniform__both") if level == 0 else (4, f"noise{level}cm")
            means.append(1000*np.sqrt(summary_value(summary, events, stage, condition, "adapted", start, "mse_h")))
            frozen.append(1000*np.sqrt(summary_value(summary, events, stage, condition, "pretrained", start, "mse_h")))
        if not np.allclose(frozen, frozen[0]):
            raise ValueError("Frozen baseline changes across noise levels")
        ax.plot(levels, means, marker=marker, color=color, label=label)
        ax.axhline(frozen[0], linestyle="--", color=color, alpha=.7, label=f"Frozen, {label}")
    ax.set(title="Observation-noise sensitivity", yscale="log",
           xlabel="Observation-noise standard deviation (cm)", xticks=levels,
           ylabel=r"Water-level RMSE (mm)")
    ax.grid(alpha=.18, which="both")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .15, 1, 1))
    save(fig, output, "noise_sensitivity_20cm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=REPO / "results/reported/sparse_adaptation")
    parser.add_argument("--output-dir", type=Path, default=REPO / "docs/figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.titlesize": 14, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10})
    summary = read_tsv(args.results_root / "summary.tsv")
    events = read_tsv(args.results_root / "event_metrics.tsv")
    jobs = read_tsv(args.results_root / "run_index.tsv")
    comparison_figure(args.output_dir, summary, events, 1,
        {"frozen": "Frozen", "full": "Full", "first_two": "First two", "lora": "LoRA",
         "last_two": "Last two", "output": "Output only", "film": "FiLM"}, "parameter_accuracy")
    comparison_figure(args.output_dir, summary, events, 2,
        {"frozen": "Frozen", **{n: label for n, label in LABELS.items() if n != "frozen"}},
        "temporal_objective_comparison")
    temporal_figures(args.results_root, args.output_dir, summary, events, jobs)
    curriculum_figure(args.results_root, args.output_dir)
    noise_figure(args.output_dir, summary, events)
    (args.output_dir / "chapter_6_figures_notes.txt").write_text(
        "Flux-correction adapter of the original chapter_6.py.\n"
        "Costs average 15 fits; errors average repeats within each of 18 held-out events.\n"
        "Figures show overall water-level RMSE only, with the square root taken after squared-error aggregation.\n"
        "parameter_accuracy and temporal_objective_comparison directly compare early/late errors.\n"
        "Lead64 uses saved step 64. Early/late averages are not exact endpoint errors.\n"
        "Exact 12-hour state interpolation and curriculum phase costs are unavailable here.\n"
        "Curriculum validation progress replaces the old phase-cost figure; timings are not invented.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
