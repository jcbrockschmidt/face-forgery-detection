#!/usr/bin/env python3

"""
Tests a model against a testing set.

```
./test.py <DATA_DIR> <CLASS> <WEIGHTS>
```

Example:

```
./test.py data/c0/test icf model/best.hdf5
```
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence Tensorflow warnings.

import argparse
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sys import stderr

from classifiers import CLASS_MODES, MODEL_MAP
from utils import create_data_generator, tnr_pred, tpr_pred

# Silence Tensorflow warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main(data_dir, other_classes, weights_path, mtype,
         class_mode='binary',
         batch_size=16):
    """
    Tests a model.

    Args:
        data_dir: Directory containing test classes, including "real"
            and `other_class`.
        other_classes: Collection of classes other than "real" to test on.
        weights_path: Path to HDF5 weights file for the model.
        class_mode: See `keras.preprocessing.image.ImageDataGenerator.flow_from_directory`.
        batch_size: Number of images to process at a time.
    """
    # Make sure classes exist
    for c in other_classes + ['real']:
        test_dir = os.path.join(data_dir, c)
        if not os.path.exists(test_dir):
            print('ERROR: "{}" has no class "{}"'.format(test_dir, c),
                  file=stderr)
            exit(2)

    # Make sure model is valid.
    if not mtype in MODEL_MAP:
        print('ERROR: "{}" is not a valid model type'.format(mtype),
              file=stderr)
        exit(2)

    # Create data generators.
    print('\nLoading testing data from "{}"...'.format(data_dir))
    test_generator, _ = create_data_generator(data_dir, other_classes, batch_size, class_mode)

    # Create model.
    model = MODEL_MAP[mtype]()
    model.load(weights_path)
    model.set_metrics(['acc', tpr_pred, tnr_pred])

    # Test model.
    classes_str = ', '.join(other_classes)
    print('\nTesting {} model on class {}...\n'.format(mtype.upper(), classes_str.upper()))
    mse, acc, tpr, tnr = model.evaluate_with_generator(test_generator)
    print('mse:\t{}'.format(mse))
    print('acc:\t{}'.format(acc))
    print('tpr:\t{}'.format(tpr))
    print('tnr:\t{}'.format(tnr))


if __name__ == '__main__':
    try:
        # Construct a string listing all model types in the format
        #
        #    'model1, model2, model3, or model4'
        #
        models = ['"{}"'.format(k.lower()) for k in MODEL_MAP.keys()]
        models_str = '{}, or {}'.format(', '.join(models[:-1]), models[-1])

        parser = argparse.ArgumentParser(
            description='Tests a model')
        parser.add_argument('-d', '--data-dir', dest='data_dir', type=str,
                            required=True, nargs=1,
                            help='directory containing a "train" and "val" directory')
        parser.add_argument('-c', '--classes', metavar='classes', dest='other_classes',
                            type=str, nargs='+', required=True,
                            help='classes other than "real" to train on')
        parser.add_argument('-w', '--weights', type=str, required=True, nargs=1,
                            default=[None],
                            help='HDF5 weight file to initialize model with')
        parser.add_argument('-m', '--mtype', type=str, required=True, nargs=1,
                            help='model type, either {}'.format(models_str))
        parser.add_argument('-cm', '--class-mode', dest='class_mode', type=str,
                            required=False, nargs=1, default=['binary'],
                            help='type of "binary" or "categorical"')
        parser.add_argument('-b', '--batch-size', metavar='batch_size', type=int,
                            required=False, nargs=1, default=[16],
                            help='number of images to read at a time')
        parser.add_argument('-g', '--gpu-fraction', metavar='batch_size', type=float,
                            required=False, nargs=1, default=[1.0],
                            help='maximum fraction of the GPU\'s memory the ' \
                            'model is allowed to use, between 0.0 and 1.0')
        args = parser.parse_args()

        data_dir = args.data_dir[0]
        other_classes = args.other_classes
        weights_path = args.weights[0]
        mtype = args.mtype[0]
        class_mode = args.class_mode[0].lower()
        batch_size = args.batch_size[0]
        gpu_frac = args.gpu_fraction[0]

        # Validate arguments.
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)
        if not weights_path is None and not os.path.isfile(weights_path):
            print('"{}" is not a file'.format(weights_path), file=stderr)
            exit(2)
        if gpu_frac < 0 or gpu_frac > 1:
            print('gpu-fraction must be between 0.0 and 1.0', file=stderr)
            exit(2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
        sess = tf.Session(config=config)
        set_session(sess)

        main(data_dir, other_classes, weights_path, mtype,
             batch_size=batch_size,
             class_mode=class_mode)

    except KeyboardInterrupt:
        print('Program terminated prematurely')
