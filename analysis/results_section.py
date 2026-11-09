"""Regenerate every G-Results figure from v4 saved evaluations, without training."""
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from results_common import REPO, LABELS, read_tsv, save, summary_value

ROOT = REPO / 'results/reported/sparse_adaptation'
OUT = REPO / 'docs/figures'

def value(summary, events, audit, stage, condition, model, start, metric):
    result = summary_value(summary, events, stage, condition, model, start, metric)
    audit.append(dict(stage=stage, condition=condition, model=model,
                      start=start, metric='rmse_h', value=1000*np.sqrt(result), unit='mm'))
    return result

def panels(summary, events, audit, output, stage, conditions, labels, filename, starts=(0.,48.)):
    fig, ax = plt.subplots(figsize=(15/2.54, .35*len(conditions)+1.4))
    for start, offset, color, marker in zip(starts, [-.12, .12], ['#0072B2', '#D55E00'], ['o', 's']):
        y = np.arange(len(conditions))+offset
        vals = [1000*np.sqrt(value(summary, events, audit, stage,c,m,start,'mse_h')) for c,m in conditions]
        ax.scatter(vals, y, s=36, color=color, marker=marker, zorder=3,
                   label=f'{start:g}–{12 if start==0 else 72} h')
    ax.set(yticks=np.arange(len(conditions)), yticklabels=labels, xscale='log',
           xlabel='Water-level RMSE (mm)')
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    ax.invert_yaxis()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.15), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig,output,filename)

def main():
    output = OUT
    output.mkdir(parents=True, exist_ok=True)
    summary = read_tsv(ROOT / 'summary.tsv')
    events = read_tsv(ROOT / 'event_metrics.tsv')
    audit = []
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    parameters = ['full','first_two','lora','last_two','output','film']
    panels(summary, events, audit, output, 1,[('full','pretrained')]+[(c,'adapted') for c in parameters],
           ['Frozen','Full','First two','LoRA','Last two','Output only','FiLM'],'parameter_accuracy_flux')
    objectives = list(LABELS)
    conditions = [('long' if c=='frozen' else c,'pretrained' if c=='frozen' else 'adapted') for c in objectives]
    # One overall water-level RMSE per objective and interval.
    panels(summary, events, audit, output, 2,conditions,[LABELS[c] for c in objectives],'temporal_tradeoff_flux')
    geometry = list(dict.fromkeys(r['condition'] for r in summary if r['stage']=='3' and r['model']=='adapted'))
    layouts = {'both':'Two-sided','one':'One-sided','eight':'Boundary-dense','spread':'Domain-spread'}
    def geometry_label(c):
        profile, layout = c.split('__')
        return f'{profile.capitalize()}: {layouts[layout]}'
    panels(summary, events, audit, output, 3,[(c,'adapted') for c in geometry],[geometry_label(c) for c in geometry], 'observation_geometry_flux')

    fig,ax=plt.subplots(figsize=(9,5.5))
    for start,color,marker,label in [(0.,'#0072B2','o','0-12 h'),(48.,'#D55E00','s','48-72 h')]:
        levels=[0,1,5,10,20]
        vals=[1000*np.sqrt(value(summary, events, audit, 3 if n==0 else 4,'uniform__both' if n==0 else f'noise{n}cm','adapted',start,'mse_h')) for n in levels]
        baseline=1000*np.sqrt(value(summary, events, audit, 3,'uniform__both','pretrained',start,'mse_h'))
        ax.plot(levels,vals,color=color,marker=marker,label=label)
        ax.axhline(baseline,color=color,linestyle='--',label=f'Frozen {label}')
    ax.set(yscale='log',xlabel='Observation-noise standard deviation (cm)',ylabel='Water-level RMSE (mm)',xticks=levels)
    ax.grid(alpha=.2)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.15,1,1)); save(fig,OUT,'noise_sensitivity_20cm')
    with (OUT/'results_figure_values.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)
    print('Verified saved aggregate values against 54 evaluations per condition.')



if __name__ == '__main__':
    main()
