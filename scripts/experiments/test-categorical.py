#!/usr/bin/env python3

"""
Tests batches of binary classifiers and their categorical counterparts.
Calculates the accuracy against all class train on for the binary classifier,
the categorical classifier, and the categorical classifier when used as a
binary classifier.  Outputs results to a CSV file

The parameter `models_dir` should point to a directory that resembles the
following:

    models_dir
    ├── binary
    │   ├── mesoinception4
    │   │   ├── f2f-df
    │   │   ├── f2f-df-fs
    │   │   ├── f2f-fs
    │   │   ├── gann-icf-x2f
    │   │   ├── gann-x2f
    │   │   ├── icf-gann
    │   │   └── ...
    │   └── ...
    └── categorical
        ├── mesoinception4
        │   ├── f2f-df
        │   ├── f2f-df-fs
        │   ├── f2f-fs
        │   ├── gann-icf-x2f
        │   ├── gann-x2f
        │   ├── icf-gann
        │   └── ...
        └── ...

The names of the deepest subdirectories show should have some matches between
the "binary" and "categorical" directories.

The headers of the CSV file are:

    mtype, classes, bin_acc, cat_acc, cat_bin_acc

where "mtype" stand for "model type", "bin_acc" for the accuracy of the binary
classifier, "cat_acc" for the accuracy of the categorical classifier, and
"cat_bin_acc" as the accuracy of the categorical classifier when treated as a
binary real-fake classifier.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence Tensorflow warnings.

import argparse
import csv
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
from sys import stderr

from classifiers import CLASS_MODES, MODEL_MAP
from utils import cat_acc_pred, create_data_generator

# Silence Tensorflow warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

HEADERS = ('mtype', 'classes', 'bin_acc', 'cat_acc', 'cat_bin_acc')

def test_binary(mtype, data_dir, weights_path, classes, batch_size):
    """
    Loads and tests a binary classifier.

    Args:
        mtype: Architecture of models to test.
        data_dir: Directory containing directories with test images for all
            classes.
        weights_path: Path to HDF5 weights for model.
        classes: List of classes other than "real" to load.
        batch_size: Number of images to process at a time.

    Returns:
        Accuracy of classifier against the testing set.
    """
    model = MODEL_MAP[mtype](class_mode='binary')
    model.load(weights_path)
    model.set_metrics(['acc'])
    gen, _ = create_data_generator(data_dir, classes, batch_size, 'binary')
    _, acc = model.evaluate_with_generator(gen)
    return acc

def test_categorical(mtype, data_dir, weights_path, classes, batch_size):
    """
    Loads and tests a categorical classifier.

    Args:
        mtype: Architecture of models to test.
        data_dir: Directory containing directories with test images for all
            classes.
        weights_path: Path to HDF5 weights for model.
        classes: List of classes other than "real" to load.
        batch_size: Number of images to process at a time.

    Returns:
        Accuracy when treated as a categorical classifier and when treated as a
        binary classifier.
    """
    num_classes = len(classes) + 1
    model = MODEL_MAP[mtype](class_mode='categorical', classes=num_classes)
    model.load(weights_path)
    model.set_metrics(['acc', cat_acc_pred])
    gen, _ = create_data_generator(data_dir, classes, batch_size, 'categorical')
    _, cat_acc, cat_bin_acc = model.evaluate_with_generator(gen)
    return cat_acc, cat_bin_acc

def main(data_dir, models_dir, mtype, output_file, batch_size=16):
    """
    Tests binary classifiers and their categorical counterparts.

    Args:
        data_dir: Directory containing directories with test images for all
            classes.
        models_dir: Models directory as described in this script's docstring.
        mtype: Architecture of models to test.
        output_file: CSV file to output to.
        batch_size: Number of images to process at a time.
    """
    # Make sure model is valid.
    if not mtype in MODEL_MAP:
        print('ERROR: "{}" is not a valid model type'.format(mtype),
              file=stderr)
        exit(2)

    print('Testing binary and categorical models for {}'.format(mtype.upper()))
    print('Loading models from "{}"'.format(models_dir))
    print('Outputting to "{}"'.format(output_file))
    print('Batch size: {}'.format(batch_size))

    # Open output file.  Initialize with headers if it does not exist.
    init_output = not os.path.exists(output_file)
    output = open(output_file, 'a')
    output_csv = csv.writer(output)
    if init_output:
        output_csv.writerow(HEADERS)

    bin_dir = os.path.join(models_dir, 'binary')
    cat_dir = os.path.join(models_dir, 'categorical')
    mtype_bin_dir = os.path.join(bin_dir, mtype.lower())
    mtype_cat_dir = os.path.join(cat_dir, mtype.lower())
    for d in (bin_dir, cat_dir, mtype_bin_dir, mtype_cat_dir):
        if not os.path.isdir(d):
            print('ERROR: Directory "{}" is missing'.format(d),
                  file=stderr)
            exit(1)
        if not os.path.isdir(cat_dir):
            print('ERROR: Directory "{}" is missing'.format(bin_dir),
                  file=stderr)
            exit(1)

    # Calculate recall for all models for all classes.
    for model_name in sorted(os.listdir(mtype_bin_dir)):
        bin_weights_dir = os.path.join(mtype_bin_dir, model_name)
        cat_weights_dir = os.path.join(mtype_cat_dir, model_name)

        # See if counterpart exists and both are directories.
        if not (os.path.isdir(bin_weights_dir) and os.path.isdir(cat_weights_dir)):
            continue

        # Make sure best weight parameters are present.
        bin_best_path = os.path.join(bin_weights_dir, 'best.hdf5')
        cat_best_path = os.path.join(cat_weights_dir, 'best.hdf5')
        for p in (bin_best_path, cat_best_path):
            if not os.path.isfile(p):
                print('ERROR: File "{}" does not exist. Skipping.'.format(p),
                      file=stderr)
                continue

        print('\nLoading data generators for "{}"'.format(model_name))

        print('\nTesting model "{}"...'.format(model_name))
        classes = model_name.split('-')

        bin_acc = test_binary(
            mtype, data_dir, bin_best_path, classes, batch_size)
        cat_acc, cat_bin_acc = test_categorical(
            mtype, data_dir, cat_best_path, classes, batch_size)

        # Write data.
        classes_str = ','.join(model_name.split('-'))
        data_line = (mtype.lower(), classes_str, bin_acc, cat_acc, cat_bin_acc)
        output_csv.writerow(data_line)
        output.flush()

    output.close()

if __name__ == '__main__':
    try:
        desc = 'Tests batches of binary classifiers and their categorical ' \
               'counterparts'
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('-d', '--data-dir', dest='data_dir', type=str,
                            required=True, nargs=1,
                            help='directory containing subdirectories for each class')
        parser.add_argument('-md', '--models_dir', type=str, required=True, nargs=1,
                            default=[None],
                            help='directory described in script description')
        parser.add_argument('-m', '--mtype', type=str, required=True, nargs=1,
                            help='model type')
        parser.add_argument('-o', '--output', dest='output_file', type=str,
                            required=True, nargs=1,
                            help='path to CSV to write or append data to')
        parser.add_argument('-b', '--batch-size', metavar='batch_size', type=int,
                            required=False, nargs=1, default=[16],
                            help='number of images to read at a time')
        parser.add_argument('-g', '--gpu-fraction', metavar='gpu_fraction', type=float,
                            required=False, nargs=1, default=[1.0],
                            help='maximum fraction of the GPU\'s memory the ' \
                            'model is allowed to use, between 0.0 and 1.0')
        args = parser.parse_args()

        data_dir = args.data_dir[0]
        models_dir = args.models_dir[0]
        mtype = args.mtype[0]
        output_file = args.output_file[0]
        batch_size = args.batch_size[0]
        gpu_frac = args.gpu_fraction[0]

        # Validate arguments.
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)
        if not os.path.isdir(models_dir):
            print('"{}" is not a directory'.format(models_dir), file=stderr)
            exit(2)
        if gpu_frac < 0 or gpu_frac > 1:
            print('gpu-fraction must be between 0.0 and 1.0', file=stderr)
            exit(2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
        sess = tf.Session(config=config)
        set_session(sess)

        main(data_dir, models_dir, mtype, output_file, batch_size=batch_size)

    except KeyboardInterrupt:
        print('Program terminated prematurely')
