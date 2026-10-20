"""Update the two active temporal plots from saved results only.

Run from the repository root: python analysis/results_revision.py
No experiments, forecasts or other thesis figures are regenerated.
"""
import numpy as np
import matplotlib.pyplot as plt

import results_section as results
from chapter_6 import LABELS, STYLES, save


def main():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9.5,
        'axes.labelsize': 9.5, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'legend.fontsize': 9,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    objectives = list(LABELS)
    conditions = [('long', 'pretrained') if key == 'frozen' else (key, 'adapted')
                  for key in objectives]
    results.panels(2, conditions, [LABELS[key] for key in objectives],
                   'temporal_tradeoff_flux')

    fig, ax = plt.subplots(figsize=(15 / 2.54, 4.2))
    values = []
    for key, (condition, model) in zip(objectives, conditions):
        fits = [job for job in results.jobs
                if job['stage'] == '2' and job['temporal'] == key]
        if key == 'frozen':
            cost = 0.0
        else:
            assert len(fits) == 15
            cost = np.mean([float(job['training_seconds']) for job in fits])
        error = 1000 * np.sqrt(results.value(2, condition, model, 0., 'mse_h'))
        values.append(error)
        ax.scatter(cost, error, **STYLES[key], s=42, label=LABELS[key])
    ax.set(yscale='log', xlabel='Mean training-loop time per fit (s)',
           ylabel='Water-level RMSE (mm)',
           ylim=(min(values) * .6, max(values) * 1.7))
    ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(), loc='lower center',
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, .19, 1, 1))
    save(fig, results.OUT, 'forecast_length_computation_cost_0to12h_flux')
    print('Verified equal-event aggregates and regenerated only the two temporal figures.')


if __name__ == '__main__':
    main()
