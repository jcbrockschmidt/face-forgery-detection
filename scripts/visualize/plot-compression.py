#!/usr/bin/env python3

"""
Plots data from the CSV file generated by `scripts/experiments/test-compression.py`
for all three compression levels.  Displays plots on screen.
"""

import argparse
import csv
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
import numpy as np
import os
from sys import stderr

from common import CLASS_TO_COLOR, CLASS_TO_LABEL

HEADERS = (
    'mtype', 'comp', 'class',
    'acc_all', 'tpr_all', 'tnr_all',
    'acc_c0', 'tpr_c0', 'tnr_c0',
    'acc_c23', 'tpr_c23', 'tnr_c23',
    'acc_c40', 'tpr_c40', 'tnr_c40'
)

HINDEX = {v: i for i, v in enumerate(HEADERS)}

CLASSES = ('f2f', 'df', 'fs', 'icf', 'gann', 'x2f')

COMP_TO_LINESTYLE = {
    'c0'  : '--',
    'c23' : '-.',
    'c40' : ':',
    'all' : '-'
}

COMP_TO_LABEL = {
    'c0'  : 'Trained on lossless',
    'c23' : 'Trained on visibly lossless',
    'c40' : 'Trained on lossy',
    'all' : 'Trained on all'
}

COMP_LEVELS = ('c0', 'c23', 'c40')

X_LABELS = ('Lossless', 'Visibly lossless', 'Lossy')

GUIDE_COLOR = (0.9, 0.9, 0.9)

def x_to_comp(tick_val, tick_pos):
    """
    A formatting function for `matplotlib.ticker.FuncFormatter`.
    """
    if tick_val >= 0 and tick_val < len(X_LABELS):
        return X_LABELS[int(tick_val)]
    else:
        return ''

def load_model_data(csv_path):
    """
    Loads model results from a CSV file created by
    `scripts/experiments/test-compression.py`.

    Args:
        csv_path: Path to CSV file.

    Returns:
        A dictionary mapping model name -> class -> trained compression level
            -> tested compression level -> {accuracy, TPR, TNR}.
    """
    models = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')

        # Skip the headers.
        next(reader)

        for row in reader:
            mtype = row[HINDEX['mtype']]
            class_type = row[HINDEX['class']]
            train_comp = row[HINDEX['comp']]

            if mtype not in models:
                models[mtype] = {}
            if class_type not in models[mtype]:
                models[mtype][class_type] = {}

            test_results = {}
            for test_comp in list(COMP_LEVELS) + ['all']:
                res = {}
                for metric in ('acc', 'tpr', 'tnr'):
                    label = '{}_{}'.format(metric, test_comp)
                    res[metric] = row[HINDEX[label]]
                test_results[test_comp] = res

            models[mtype][class_type][train_comp] = test_results

    return models

def plot_accuracy(models):
    """
    Plots accuracies for each compression level.

    Args:
        models: A dictionary returned by `load_model_data`.

    Returns:
        Matplotlib figure and axes.
    """
    # Plot accuracy of models against each compression level.
    fig, ax = plt.subplots()
    ax.set_xlabel('Compression level')
    ax.set_ylabel('Accuracy')
    ax.xaxis.set_major_formatter(FuncFormatter(x_to_comp))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    xs = np.arange(len(COMP_LEVELS))
    leg_handles = []
    leg_labels = []
    for mtype in models:
        model = models[mtype]
        for class_type in CLASSES:
            if class_type not in model:
                continue
            color = CLASS_TO_COLOR[class_type]
            for train_comp in list(COMP_LEVELS) + ['all']:
                res = model[class_type][train_comp]
                label = '{}, {}'.format(class_type, train_comp)
                ys = (res['c0']['acc'], res['c23']['acc'], res['c40']['acc'])
                ys = [float(y) for y in ys]
                linstyle = COMP_TO_LINESTYLE[train_comp]
                ax.plot(xs, ys, linstyle, label=label, color=color)

            # Create custom legend entries.  Only want to show one entry
            # for each class type.
            leg_handles.append(Line2D([], [], linestyle='-', color=color))
            leg_labels.append(CLASS_TO_LABEL[class_type])

    # Draw lines marking each compression level.

    # Add guidelines for the X ticks.
    ax.set_axisbelow(True)
    ax.xaxis.grid(color=GUIDE_COLOR, linestyle='-')

    # Add legend entries for line styles.
    for comp in COMP_TO_LINESTYLE:
        linstyle = COMP_TO_LINESTYLE[comp]
        leg_handles.append(Line2D([], [], linestyle=linstyle, color=(0, 0, 0)))
        leg_labels.append(COMP_TO_LABEL[comp])

    ax.legend(leg_handles, leg_labels)

    return fig, ax

def plot_tpr_vs_tnr(models):
    """
    Plots the ratios of true positive rates to true negative rates for each
    compression level.

    Args:
        models: A dictionary returned by `load_model_data`.

    Returns:
        Matplotlib figure and axes.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('Compression level')
    ax.set_ylabel('TPR to TNR ratio')
    ax.xaxis.set_major_formatter(FuncFormatter(x_to_comp))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale('log')
    ax.tick_params(axis='y', which='minor')
    ax.tick_params(axis='y', which='major')
    ax.yaxis.set_minor_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    xs = np.arange(len(COMP_LEVELS))
    leg_handles = []
    leg_labels = []
    for mtype in models:
        model = models[mtype]
        for class_type in CLASSES:
            if class_type not in model:
                continue
            color = CLASS_TO_COLOR[class_type]
            for train_comp in list(COMP_LEVELS) + ['all']:
                res = model[class_type][train_comp]
                tprs = (res['c0']['tpr'], res['c23']['tpr'], res['c40']['tpr'])
                tnrs = (res['c0']['tnr'], res['c23']['tnr'], res['c40']['tnr'])
                ratios = [float(tprs[i]) / float(tnrs[i]) for i in range(len(tprs))]
                linstyle = COMP_TO_LINESTYLE[train_comp]
                ax.plot(xs, ratios, linstyle, color=color)

            # Create custom legend entries.  Only want to show one entry
            # for each class type.
            leg_handles.append(Line2D([], [], linestyle='-', color=color))
            leg_labels.append(CLASS_TO_LABEL[class_type])

    # Add guidelines for the X ticks.
    ax.set_axisbelow(True)
    ax.xaxis.grid(color=GUIDE_COLOR, linestyle='-')

    # Add a guideline at y = 1.
    ax.set_axisbelow(True)
    plt.axhline(y=1, linestyle='-', color=GUIDE_COLOR)
    '''
    for x in xs:
        plt.axvline(x=x, linestyle='-', color=GUIDE_COLOR)
    plt.axhline(y=1, linestyle='-', color=GUIDE_COLOR)
    '''

    # Add legend entries for line styles.
    for comp in COMP_TO_LINESTYLE:
        linstyle = COMP_TO_LINESTYLE[comp]
        leg_handles.append(Line2D([], [], linestyle=linstyle, color=(0, 0, 0)))
        leg_labels.append(COMP_TO_LABEL[comp])

    # Hide every other minor tick label.
    for label in ax.get_yticklabels(which='minor')[1::2]:
        label.set_visible(False)

    ax.legend(leg_handles, leg_labels)

    return fig, ax

def plot_combined(models):
    """
    Plots a bar graph with accuracies for models trained on all compression levels.

    Args:
        models: A dictionary returned by `load_model_data`.

    Returns:
        Matplotlib figure and axes.
    """
    BAR_WIDTH = 0.2
    METRICS = ('acc', 'tpr', 'tnr')
    METRIC_TO_COLOR = {
        'acc' : (0.5, 0, 0.5),  # Purple
        'tpr' : (1, 0, 0),      # Red
        'tnr' : (0, 0, 1)       # Blue
    }
    METRIC_TO_LABEL = {
        'acc' : 'Accuracy',
        'tpr' : 'Real recall',
        'tnr' : 'Fake recall'
    }

    fig, ax = plt.subplots()
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    y_min = 1.0
    xs = np.arange(len(CLASSES)) - BAR_WIDTH
    for mtype in models:
        for i, metric in enumerate(METRICS):
            ys = []
            for c in CLASSES:
                res = models[mtype][c]['all']['all']
                m = float(res[metric])
                ys.append(m)
                y_min = min(y_min, m)
            ax.bar(xs + (BAR_WIDTH * i), ys, BAR_WIDTH,
                   color=METRIC_TO_COLOR[metric],
                   label=METRIC_TO_LABEL[metric])

    x_labels = [CLASS_TO_LABEL[c] for c in CLASSES]
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(x_labels)
    ax.set_ylim((y_min - 0.05, 1.005))

    # Add guidelines for the Y ticks.
    ax.set_axisbelow(True)
    ax.yaxis.grid(color=GUIDE_COLOR, linestyle='-')

    ax.legend()

    return fig, ax

def main(csv_path):
    """
    Creates and displays various plots using the supplied experimentation data.
    """
    models = load_model_data(csv_path)

    # Plot accuracies against each compression level.
    plot_accuracy(models)

    # Plot TPR vs. TNR of models against each compression level.
    plot_tpr_vs_tnr(models)

    # Plot accuracy, TPR, and TNR for models trained on all compression levels.
    plot_combined(models)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests a model')
    parser.add_argument('input', type=str, nargs=1,
                        help='path to CSV file with compression data')
    args = parser.parse_args()

    csv_path = args.input[0]

    if not os.path.isfile(csv_path):
        print('"{}" is not a file'.format(csv_path), file=stderr)
        exit(2)

    main(csv_path)
