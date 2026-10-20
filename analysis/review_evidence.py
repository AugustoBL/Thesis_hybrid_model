"""Plot supplementary evidence from saved errors only; no model execution.

Run from the repository root: python analysis/review_evidence.py
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import results_section as figures
from chapter_6 import expected_keys, local_run, save


def write_rows(name, rows):
    with (figures.OUT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def event_rmse(stage, condition, start):
    rows = [r for r in figures.events if int(r['stage']) == stage
            and r['condition'] == condition and r['model'] == 'adapted'
            and float(r['from_hours']) == start]
    keys = [(int(r['fold']), int(r['repeat']), int(r['event'])) for r in rows]
    assert len(keys) == 54 and set(keys) == expected_keys()
    return np.array([np.sqrt(np.mean([float(r['mse_h']) for r in rows
                                     if int(r['event']) == event]))
                     for event in range(1, 19)])


def paired_events():
    fig, axes = plt.subplots(1, 2, figsize=(15/2.54, 5.2), sharey=True)
    rows = []
    for ax, (stage, first, second, title) in zip(axes, [
            (1, 'lora', 'full', 'LoRA versus full'),
            (2, 'balanced', 'long', 'Multi-step versus long-128')]):
        for start, color, marker, label in [(0., '#0072B2', 'o', '0-12 h'),
                                           (48., '#D55E00', 's', '48-72 h')]:
            change = 100 * (event_rmse(stage, first, start) /
                            event_rmse(stage, second, start) - 1)
            ax.scatter(change, np.arange(1, 19), c=color, marker=marker, label=label, s=30)
            rows.extend(dict(comparison=f'{first}_vs_{second}', event=event,
                             from_hours=start, rmse_change_percent=float(delta))
                        for event, delta in enumerate(change, 1))
        ax.axvline(0, color='#555555', lw=1)
        ax.set(title=title, xlabel='RMSE change (%)', yticks=np.arange(1, 19))
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Held-out event')
    axes[0].invert_yaxis()
    fig.legend(*axes[1].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save(fig, figures.OUT, 'paired_event_rmse_changes')
    write_rows('paired_event_rmse_changes.csv', rows)


def observation_history():
    jobs = [j for j in figures.jobs if int(j['stage']) == 1 and j['method'] == 'full']
    assert len(jobs) == 15
    means = {}
    reference_steps = None
    for model in ('pretrained', 'adapted'):
        arrays, keys = [], []
        for job in jobs:
            fold = int(job['fold'])
            for event in range((fold-1)*18//5+1, fold*18//5+1):
                path = local_run(figures.ROOT, job) / 'diagnostics' / f'{model}_event_{event}.tsv'
                array = np.loadtxt(path, delimiter='\t', skiprows=1, usecols=(0, 6, 7))
                if reference_steps is None:
                    reference_steps = array[:, 0]
                assert np.array_equal(reference_steps, array[:, 0])
                arrays.append(array[:, 1:])
                keys.append((fold, int(job['repeat']), event))
        assert len(keys) == 54 and set(keys) == expected_keys()
        arrays = np.stack(arrays)
        event_means = [arrays[[key[2] == event for key in keys]].mean(axis=0)
                       for event in range(1, 19)]
        means[model] = 1000 * np.sqrt(np.mean(event_means, axis=0))
    grid = np.arange(101) * 1000.
    depth = 12+8*(1+np.tanh((grid-30000)/8000))-8*(1+np.tanh((grid-70000)/8000))
    hours = reference_steps * (.7*1000/np.sqrt(depth.max()*9.81)) / 3600
    fig, axes = plt.subplots(2, 1, figsize=(15/2.54, 5.2), sharex=True, sharey=True)
    for column, (ax, title) in enumerate(zip(axes, ['Observed gauges', 'Unobserved interior (cells 9–92)'])):
        for model, label, color in [('pretrained', 'Frozen', '#666666'),
                                     ('adapted', 'Full fine-tuning', '#0072B2')]:
            ax.plot(hours[::8], means[model][::8, column], color=color, label=label, lw=1.4)
        ax.axvspan(0, 12, color='#eeeeee', zorder=0)
        ax.set(title=title, xlim=(0, 72), ylabel='Water-level RMSE (mm)',
               yscale='log', xticks=np.arange(0, 73, 12))
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, loc='lower right', ncol=2)
    axes[1].set_xlabel('Forecast lead (h)')
    fig.tight_layout()
    save(fig, figures.OUT, 'observed_unobserved_rmse')
    write_rows('observed_unobserved_rmse.csv', [
        dict(lead_hours=float(t), frozen_gauges_mm=float(means['pretrained'][i, 0]),
             adapted_gauges_mm=float(means['adapted'][i, 0]),
             frozen_interior_mm=float(means['pretrained'][i, 1]),
             adapted_interior_mm=float(means['adapted'][i, 1]))
        for i, t in enumerate(hours)])


def main():
    parameters = ['full', 'first_two', 'lora', 'last_two', 'output', 'film']
    figures.panels(1, [('full', 'pretrained')]+[(c, 'adapted') for c in parameters],
                   ['Frozen', 'Full', 'First two', 'LoRA', 'Last two', 'Output only', 'FiLM'],
                   'parameter_accuracy_flux')
    objectives = list(figures.LABELS)
    conditions = [('long' if c == 'frozen' else c, 'pretrained' if c == 'frozen' else 'adapted')
                  for c in objectives]
    figures.panels(2, conditions, [figures.LABELS[c] for c in objectives], 'temporal_tradeoff_flux')
    geometry = list(dict.fromkeys(r['condition'] for r in figures.summary
                                 if r['stage'] == '3' and r['model'] == 'adapted'))
    layouts = {'both': 'Two-sided', 'one': 'One-sided', 'eight': 'Boundary-dense', 'spread': 'Domain-spread'}
    labels = [f'{c.split("__")[0].capitalize()}: {layouts[c.split("__")[1]]}' for c in geometry]
    figures.panels(3, [(c, 'adapted') for c in geometry], labels, 'observation_geometry_flux')
    paired_events()
    observation_history()


if __name__ == '__main__':
    main()
