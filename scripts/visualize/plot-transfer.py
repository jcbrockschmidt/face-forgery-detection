#!/usr/bin/env python3

"""
Plots data from the CSV file generated by `scripts/experiments/test-transfer.py`.
Creates a heatmap of the accuracies for each model against the class transferred to.
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from sys import stderr

from common import CLASS_TO_LABEL

HEADERS = (
    'mtype', 'orig_class', 'trans_class',
    'real', 'df', 'f2f', 'fs', 'gann', 'icf', 'x2f'
)

HINDEX = {v: i for i, v in enumerate(HEADERS)}

CLASSES = ('x2f', 'gann', 'icf', 'fs', 'f2f', 'df')

def load_model_data(csv_path):
    """
    Loads model results from a CSV file created by
    `scripts/experiments/test-transfer.py`.

    Args:
        csv_path: Path to CSV file.

    Returns:
        A dictionary mapping model name -> original class -> transferred class
            -> tested class -> accuracy
    """
    models = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')

        # Skip the headers.
        next(reader)

        for row in reader:
            mtype = row[HINDEX['mtype']]
            orig_class = row[HINDEX['orig_class']]
            trans_class = row[HINDEX['trans_class']]

            if mtype not in models:
                models[mtype] = {}
            if orig_class not in models[mtype]:
                models[mtype][orig_class] = {}

            test_results = {}
            for test_class in CLASSES:
                test_results[test_class] = row[HINDEX[test_class]]

            models[mtype][orig_class][trans_class] = test_results

    return models

def plot_heatmap(models):
    """
    Creates a heatmap of the accuracies for each model against the class transferred to.
    """
    fig, ax = plt.subplots()
    ax.set_xlabel('Class Retrained On')
    ax.set_ylabel('Original Class')

    z = []
    for m in models:
        for row, orig_class in enumerate(CLASSES):
            for col, trans_class in enumerate(CLASSES):
                if orig_class == trans_class:
                    acc = 0
                else:
                    acc = float(models[m][orig_class][trans_class][trans_class])
                z.append(acc)

    z = np.reshape(np.array(z), (len(CLASSES), len(CLASSES)))
    heatmap = ax.pcolor(z, cmap=plt.get_cmap('Purples'))
    fig.colorbar(heatmap)

    # Add class labels.
    labels = [CLASS_TO_LABEL[c] for c in CLASSES]
    ax.set_xticks(np.arange(0, len(CLASSES)) + 0.5)
    ax.set_yticks(np.arange(0, len(CLASSES)) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=45)

    # Make more room for labels.
    fig.subplots_adjust(left=0.2, bottom=0.25)

    # Write accuracies in each cell.
    heatmap.update_scalarmappable()
    for p, acc in zip(heatmap.get_paths(), heatmap.get_array()):
        x, y = p.vertices[:-1, :].mean(0)
        ax.text(x + 0.1, y, '%.3f' % acc, ha='center', color=(1, 1, 1))

    # Remove ticks and plot outline.
    ax.tick_params(top=False, bottom=False, left=False, right=False)
    for spine in fig.gca().spines.values():
        spine.set_visible(False)

    return fig, ax

def main(csv_path):
    """
    Creates and displays heatmap using the supplied experimentation data.
    """
    models = load_model_data(csv_path)

    plot_heatmap(models)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots transfer learning data')
    parser.add_argument('input', type=str, nargs=1,
                        help='path to CSV file with transfer learning data')
    args = parser.parse_args()

    csv_path = args.input[0]

    if not os.path.isfile(csv_path):
        print('"{}" is not a file'.format(csv_path), file=stderr)
        exit(2)

    main(csv_path)
