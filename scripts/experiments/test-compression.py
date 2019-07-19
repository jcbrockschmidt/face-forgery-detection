#!/usr/bin/env python3

"""
Tests a batch of models trained on different compression levels on every
(available) compression level and outputs the results to a CSV file.

The parameter `models_dir` should point to a directory that resembles the
following:

    models_dir
    ├── all
    │   ├── mesoinception4
    │   │   ├── df
    │   │   │   └── best.hdf5
    │   │   ├── f2f
    │   │   │   └── best.hdf5
    │   │   ├── fs
    │   │   │   └── best.hdf5
    │   │   ├── gann
    │   │   │   └── best.hdf5
    │   │   ├── icf
    │   │   │   └── best.hdf5
    │   │   └── x2f
    │   │       └── best.hdf5
    │   └── ...
    ├── c0
    │   └── ...
    ├── c23
    │   └── ...
    └── c40
        └── ...

The headers of the CSV file are:

    mtype, comp, class, acc_all, tpr_all, tnr_all, acc_c0, tpr_c0, tnr_c0,
    acc_c23, tpr_c23, tnr_c23, acc_c40, tpr_c40, tnr_c40

where "mtype" stand for "model type", "comp" for "compression", "acc" for
"accuracy", "tpr" for "true positive rate" (recall for real faces), and "tnr"
for "true negative rate" (recall for fake faces).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence Tensorflow warnings.

import argparse
import csv
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sys import stderr

from classifiers import CLASS_MODES, MODEL_MAP
from utils import create_data_generator, tnr_pred, tpr_pred

# Silence Tensorflow warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main(data_dir, models_dir, mtype, output_file, batch_size=16):
    """
    Tests models on every available compression level.

    Args:
        data_dir: Directory containing test classes, including "real"
            and `other_class`.
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

    print('Testing compression levels for {}'.format(mtype.upper()))
    print('Loading models from "{}"'.format(models_dir))
    print('Outputting to "{}"'.format(output_file))
    print('Batch size: {}'.format(batch_size))

    # Open output file.  Initialize with headers if it does not exist.
    init_output = not os.path.exists(output_file)
    output = open(output_file, 'a')
    output_csv = csv.writer(output)
    if init_output:
        headers = (
            'mtype', 'comp', 'class',
            'acc_all', 'tpr_all', 'tnr_all',
            'acc_c0', 'tpr_c0', 'tnr_c0',
            'acc_c23', 'tpr_c23', 'tnr_c23',
            'acc_c40', 'tpr_c40', 'tnr_c40'
        )
        output_csv.writerow(headers)

    comps = ('all', 'c0', 'c23', 'c40')

    # Maps class types to maps of compression levels to data generators.
    # I.e. data generators for every class type for every compression level.
    generators = {}
    
    for comp_level in comps:
        print('\nRunning tests for compression level "{}"'.format(comp_level))
        d = os.path.join(os.path.join(models_dir, comp_level, mtype.lower()))
        if not os.path.exists(d):
            print('ERROR: "{}" does not exist. Skipping.'.format(d), file=stderr)
            continue

        comp_dir = os.path.join(models_dir, comp_level, mtype.lower())
        for md in sorted(os.listdir(comp_dir)):
            weights_dir = os.path.join(comp_dir, md)

            # Ignore files.
            if not os.path.isdir(weights_dir):
                continue

            # Make sure best weight parameters are present.
            best_path = os.path.join(weights_dir, 'best.hdf5')
            if not os.path.isfile(best_path):
                print('ERROR: File "{}" does not exist. Skipping.'.format(best_path),
                      file=stderr)
                continue

            # Create data generators if they do not exist.
            class_type = os.path.basename(md)
            if not class_type in generators:
                print('\nCreating generators for class "{}"...'.format(class_type))
                gens = {}
                gen_fail = False
                for comp_level_gen in comps:
                    test_dir = os.path.join(data_dir, comp_level_gen, 'test')
                    new_gen, _ = create_data_generator(test_dir, [class_type], batch_size, 'binary')
                    if new_gen.samples < 10:
                        print(
                            'ERROR: Only found {} samples for class ' \
                            '"{}" and "real" for compression level "{}" in "{}".'.format(
                                new_gen.samples, class_type, comp_level_gen, test_dir),
                            file=stderr
                        )
                        print('Skipping.', file=stderr)
                        gen_fail = True
                        break
                    gens[comp_level_gen] = new_gen
                    if gen_fail:
                        continue
                generators[class_type] = gens

            print('\nTesting class "{}" for compression level "{}"...'.format(
                class_type, comp_level))

            # Load model.
            model = MODEL_MAP[mtype]()
            model.load(best_path)
            model.set_metrics(['acc', tpr_pred, tnr_pred])

            # Test model on every compression level.
            results = {}
            for comp_level_test in comps:
                gen = generators[class_type][comp_level_test]
                gen.reset()
                res = {}
                _, res['acc'], res['tpr'], res['tnr'] = model.evaluate_with_generator(gen)
                results[comp_level_test] = res

            # Write data.
            data_line = (
                mtype.lower(), comp_level, class_type,
                results['all']['acc'], results['all']['tpr'], results['all']['tnr'],
                results['c0']['acc'], results['c0']['tpr'], results['c0']['tnr'],
                results['c23']['acc'], results['c23']['tpr'], results['c23']['tnr'],
                results['c40']['acc'], results['c40']['tpr'], results['c40']['tnr']
            )
            output_csv.writerow(data_line)
            output.flush()

    output.close()

if __name__ == '__main__':
    try:
        # Construct a string listing all model types in the format
        #
        #    'model1, model2, model3, or model4'
        #
        parser = argparse.ArgumentParser(
            description='Tests a model')
        parser.add_argument('-d', '--data-dir', dest='data_dir', type=str,
                            required=True, nargs=1,
                            help='directory containing directories "all", "c0", "c23", and "c40"')
        parser.add_argument('-md', '--models_dir', type=str, required=True, nargs=1,
                            default=[None],
                            help='base directory for models')
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
