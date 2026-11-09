"""Shared saved-result readers, equal-event reductions, styles, and figure output.

Importing this module does not read reports or create output directories.
"""
import csv
from io import BytesIO
import os
from pathlib import Path
import tempfile
import time

import matplotlib.pyplot as plt
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


def local_run(root, job):
    # Saved paths refer to the training device. Reconstruct from scientific keys.
    name = (f'{job["method"]}__{job["temporal"]}__{job["profile"]}__{job["layout"]}'
            f'__noise{round(100*float(job["noise_std"]))}cm')
    return root / "runs" / name / f'fold_{job["fold"]}' / f'repeat_{job["repeat"]}'
