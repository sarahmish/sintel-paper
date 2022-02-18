#!/usr/bin/env python3

"""
Generate the figures and results for the paper: "Sintel: A Machine 
Learning Framework to Extract Insights from Signals."
"""

import os
import json
import logging
import pickle
import warnings
from functools import partial
from pathlib import Path

import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from orion.benchmark import benchmark
from orion.evaluation import CONTEXTUAL_METRICS as METRICS
from orion.evaluation import contextual_confusion_matrix

from sintel.benchmark import tune_benchmark

warnings.simplefilter('ignore')

LOGGER = logging.getLogger(__name__)

plt.style.use('default')
mpl.rcParams['hatch.linewidth'] = 0.2

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data'
)
RESULT_PATH = os.path.join(DATA_PATH, '{}.csv')
OUTPUT_PATH = Path('output')
os.makedirs(OUTPUT_PATH, exist_ok=True)

PIPELINES = [
    'arima', 
    'lstm_dynamic_threshold', 
    'lstm_autoencoder', 
    'dense_autoencoder',
    'tadgan'
] 

with open(os.path.join(DATA_PATH, 'data.json'), 'r') as fp:
    DATASETS = json.load(fp)

_ORDER = ['arima', 'lstm_autoencoder', 'lstm_dynamic_threshold', 'dense_autoencoder', 'tadgan']
_LABELS = ['ARIMA', 'LSTM AE', 'LSTM DT', 'Dense AE', 'TadGAN']
_COLORS = ["#ED553B", "#F6D55C", "#3CAEA3", "#20639B", "#173F5F"]
_HATCHES = ['-', '//', '|', '\\', 'x'] * 2
_PALETTE = sns.color_palette(_COLORS)

# ------------------------------------------------------------------------------
# Saving results
# ------------------------------------------------------------------------------

def _savefig(fig, name, figdir=OUTPUT_PATH):
    figdir = Path(figdir)
    for ext in ['.png', '.pdf', '.eps']:
        fig.savefig(figdir.joinpath(name+ext),
                    bbox_inches='tight', pad_inches=0)

# ------------------------------------------------------------------------------
# Plotting results
# ------------------------------------------------------------------------------

def _get_summary(result):
    order = [
        'lstm_dynamic_threshold', 
        'tadgan', 
        'lstm_autoencoder', 
        'arima', 
        'dense_autoencoder', 
        'azure'
    ]

    family = {
        "MSL": "NASA",
        "SMAP": "NASA",
        "YAHOOA1": "YAHOO",
        "YAHOOA2": "YAHOO",
        "YAHOOA3": "YAHOO",
        "YAHOOA4": "YAHOO",
        "artificialWithAnomaly": "NAB",
        "realAWSCloudwatch": "NAB",
        "realAdExchange": "NAB",
        "realTraffic": "NAB",
        "realTweets": "NAB"
    }

    result['group'] = result['dataset'].apply(family.get)
    df = result.groupby(['group', 'dataset', 'pipeline'])[['fp', 'fn', 'tp']].sum().reset_index()

    df['precision'] = df.eval('tp / (tp + fp)')
    df['recall'] = df.eval('tp / (tp + fn)')
    df['f1'] = df.eval('2 * (precision * recall) / (precision + recall)')
    
    df = df.reset_index()
    df = df.groupby(['group', 'pipeline']).mean().reset_index()
    df = df.set_index(['group', 'pipeline'])[['f1', 'precision', 'recall']].unstack().T.reset_index(level=0)
    df = df.pivot(columns='level_0')
    df.columns = df.columns.rename('dataset', level=0)
    df.columns = df.columns.rename('metric', level=1)
    
    return df.loc[order]


def make_table_2():
    result = pd.read_csv(RESULT_PATH.format('results'))
    return _get_summary(result)


def make_figure_7a():
    profiles = pd.read_csv(RESULT_PATH.format('comp_performance'))

    profiles['pipeline'] = pd.Categorical(profiles['pipeline'], _ORDER)
    profiles = profiles.sort_values('pipeline')

    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(6.5, 2),
        gridspec_kw={'width_ratios':[1,2]}
    )
       
    memory = profiles[profiles['source'] == 'memory']
    g = sns.barplot(x="source", y='value', hue="pipeline", ax=axes[0], palette=_PALETTE,
                    data=memory, saturation=0.7, linewidth=0.5, edgecolor='k')

    for i,thisbar in enumerate(g.patches):
        thisbar.set_hatch(_HATCHES[i] * 3)
        
    time = pd.concat([
        profiles[profiles['source'] == 'fit_time'], 
        profiles[profiles['source'] == 'predict_time']
    ])
    g = sns.barplot(x="source", y='value', hue="pipeline", ax=axes[1], palette=_PALETTE,
                    data=time, saturation=0.7, linewidth=0.5, edgecolor='k')

    for bars, hatch in zip(g.containers, _HATCHES):
        for bar in bars:
            bar.set_hatch(hatch * 3)
        
    xlabels = [['Memory'], ['Training Time', 'Pipeline Latency']]

    for i in range(2):
        axes[i].set_yscale('log')
        axes[i].grid(True, linestyle='--')
        axes[i].set_xticklabels(xlabels[i])
        axes[i].get_legend().remove()
        axes[i].set_xlabel('')
          
    axes[0].set_ylim([0.2e6, 0.2e9])
    axes[1].set_ylim([0.2e1, 0.2e5])
    axes[0].set_ylabel('memory in KB (log)')
    axes[1].set_ylabel('time in seconds (log)')

    handles = [mpatches.Patch(facecolor=_COLORS[i], label=_LABELS[i], hatch=_HATCHES[i] * 3, ec='k', lw=0.5) 
               for i in range(len(_LABELS))]
    fig.legend(handles=handles, edgecolor='k', loc='upper center', bbox_to_anchor=(0.53, 1.12), ncol=len(_LABELS))
    plt.tight_layout()

    _savefig(fig, 'figure7a', figdir=OUTPUT_PATH)


def make_figure_7b():
    time = pd.read_csv(RESULT_PATH.format('delta'))

    end = time.groupby('pipeline')['end-to-end'].mean()
    alone = time.groupby('pipeline')['stand-alone'].mean()

    avg_inc = (end - alone) / alone * 100
    avg_inc = avg_inc.reset_index()
    avg_inc.columns = ['pipeline', 'inc']

    fig = plt.figure(figsize=(3.6, 3))
    ax = plt.gca()

    g = sns.barplot(x='pipeline', y='inc', data=avg_inc, palette=_PALETTE, order=_ORDER,
                linewidth=0.5, edgecolor='k', ax=ax)
    
    for i,thisbar in enumerate(g.patches):
        thisbar.set_hatch(_HATCHES[i] * 3)
        
    labels = ['\n'.join(label.split(' ')) for label in _LABELS]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    plt.title('Increase of Pipeline Runtime')
    plt.xlabel('')
    plt.ylabel('Average Increase (%)')
    plt.ylim([0, 4])
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    _savefig(fig, 'figure7b', figdir=OUTPUT_PATH)


def _get_f1_score(df):
    df = df.groupby(['dataset', 'pipeline'])[['fp', 'fn', 'tp']].sum().reset_index()

    df['precision'] = df.eval('tp / (tp + fp)')
    df['recall'] = df.eval('tp / (tp + fn)')
    df['f1'] = df.eval('2 * (precision * recall) / (precision + recall)')
    
    df = df.set_index(['dataset', 'pipeline'])[['f1']].unstack().T.droplevel(0)
    df = df.mean(axis=1).reset_index()
    df.columns = ['pipeline', 'f1']
    return df

def make_figure_7c():
    untuned = pd.read_csv(RESULT_PATH.format('untuned_results'))
    tuned = pd.read_csv(RESULT_PATH.format('tuned_results'))

    untuned = _get_f1_score(untuned)
    untuned['source'] = ['non-tuned'] * len(untuned)

    tuned = _get_f1_score(tuned)
    tuned['source'] = ['tuned'] * len(tuned)

    df = pd.concat([untuned, tuned])

    colors = ["#8d96a3", "#2e4057"]
    palette = sns.color_palette(colors)

    labels = _LABELS[1:]
    orders = _ORDER[1:]

    fig = plt.figure(figsize=(3.7, 3))

    sns.barplot(data=df, x='pipeline', y='f1', hue='source', palette=palette, edgecolor='k',
                order=orders)

    ax = plt.gca()
    for p in ax.patches:
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='k', xytext=(0, 20),
                    textcoords='offset points', rotation=90)
        
    plt.ylim([0.5, 0.8])
    plt.xticks(range(len(labels)), labels)
    plt.ylabel('F1 Score')
    plt.xlabel('')
    plt.title('Auto-Tuning F1 Score on NAB', size=13)
    plt.legend(edgecolor='k')
    plt.tight_layout()
    
    _savefig(fig, 'figure7c', figdir=OUTPUT_PATH)


def make_figure_8a():
    result = pd.read_csv(RESULT_PATH.format('results'))
    result = _get_summary(result)

    with open(os.path.join(DATA_PATH, 'semi-model.pkl'), 'rb') as f:
        scores = pickle.load(f)

    fig = plt.figure(figsize=(3.7, 3.5))
    for j, score in enumerate(scores.values()):
        plt.plot([i*2 for i in range(20)], score[:20], label=_LABELS[j], color=_COLORS[j])
        
    plt.axhline(result.loc['lstm_autoencoder']['NAB']['f1'], ls='--', c='r')
    plt.text(-1, 0.7, "best\nunsupervised", c='r', fontsize=8)

    plt.ylabel('F1 Score')
    plt.xlabel('Iteration')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), frameon=False, ncol=3)
    plt.tight_layout()

    _savefig(fig, 'figure8a', figdir=OUTPUT_PATH)


# ------------------------------------------------------------------------------
# Running benchmark
# ------------------------------------------------------------------------------

def run_benchmark():
    workers = 1 # int or "dask"

    # path of results
    result_path = os.path.join(OUTPUT_PATH, 'benchmark.csv')

    # path to save pipelines
    pipeline_dir = os.path.join(OUTPUT_PATH, 'save_pipelines')

    # path to save output on the fly
    cache_dir = os.path.join(OUTPUT_PATH, 'cache')

    # metrics
    del METRICS['accuracy']
    METRICS['confusion_matrix'] = contextual_confusion_matrix
    metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

    results = benchmark(
        pipelines=PIPELINES, datasets=DATASETS, metrics=metrics, output_path=result_path, 
        workers=workers, show_progress=True, pipeline_dir=pipeline_dir, cache_dir=cache_dir)

    return results

def run_tune_benchmark():
    workers = 1 # int or "dask"

    # path of results
    result_path = os.path.join(OUTPUT_PATH, 'tune_benchmark.csv')

    # path to save pipelines
    pipeline_dir = os.path.join(OUTPUT_PATH, 'tune_save_pipelines')

    # path to save output on the fly
    cache_dir = os.path.join(OUTPUT_PATH, 'tune_cache')

    # metrics
    del METRICS['accuracy']
    METRICS['confusion_matrix'] = contextual_confusion_matrix
    metrics = {k: partial(fun, weighted=False) for k, fun in METRICS.items()}

    # pipelines
    pipelines = ['lstm_dynamic_threshold', 'tadgan', 'lstm_autoencoder', 'dense_autoencoder']

    # datasets
    datasets = {
        "artificialWithAnomaly": BENCHMARK_DATA["artificialWithAnomaly"],
        "realAWSCloudwatch": BENCHMARK_DATA["realAWSCloudwatch"],
        "realAdExchange": BENCHMARK_DATA["realAdExchange"],
        "realTraffic": BENCHMARK_DATA["realTraffic"],
        "realTweets": BENCHMARK_DATA["realTweets"]
    }

    results = tune_benchmark(datasets=datasets,
        pipelines=pipelines, metrics=metrics, output_path=result_path, workers=workers,
        show_progress=True, pipeline_dir=pipeline_dir, cache_dir=cache_dir)

    return results

if __name__ == '__main__':
    # print("Running benchmark.. ")
    # run_benchmark()
    print("Running tuning benchmark.. ")
    run_tune_benchmark()
