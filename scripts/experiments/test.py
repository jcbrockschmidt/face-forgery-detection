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
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sys import stderr

from classifiers import MODEL_MAP

# Silence Tensorflow warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def main(data_dir, other_class, weights_path, mtype, batch_size=16):
    """
    Tests a model.

    Args:
        data_dir: Directory containing test classes, including "real"
            and `other_class`.
        other_class: Class other than "real" to test on.
        weights_path: Path to HDF5 weights file for the model.
        batch_size: Number of images to process at a time.
    """
    # Make sure classes exist
    real_dir = os.path.join(data_dir, 'real')
    other_dir = os.path.join(data_dir, other_class)
    if not os.path.exists(real_dir):
        print('ERROR: "{}" has no class "real"'.format(real_dir),
              file=stderr)
        exit(2)
    if not os.path.exists(other_dir):
        print('ERROR: "{}" has no class "{}"'.format(data_dir, other_class),
              file=stderr)
        exit(2)

    # Make sure model is valid.
    if not mtype in MODEL_MAP:
        print('ERROR: "{}" is not a valid model type'.format(mtype),
              file=stderr)
        exit(2)

    # Create data generators.
    print('\nLoading testing data from "{}"...'.format(data_dir))
    test_data_generator = ImageDataGenerator(rescale=1/255)
    test_generator = test_data_generator.flow_from_directory(
        data_dir,
        classes=[other_class, 'real'],
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    # Create model.
    model = MODEL_MAP[mtype]()
    model.load(weights_path)

    # Test model.
    print('\nTesting {} model on class {}...\n'.format(mtype.upper(), other_class.upper()))
    loss, acc = model.evaluate_with_generator(test_generator)
    print('loss: {}'.format(loss))
    print('accuracy: {}'.format(acc))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Tests a model')
        parser.add_argument('data_dir', type=str, nargs=1,
                            help='directory containing test classes')
        parser.add_argument('other_class', metavar='class', type=str, nargs=1,
                            help='class other than "real" to test on')
        parser.add_argument('weights', type=str, nargs=1,
                            help='HDF5 weight file to initialize model with')
        parser.add_argument('mtype', type=str, nargs=1,
                            help='model type, either "meso1", "meso4", ' \
                            '"mesoinception4", "mesoinc4frozen16", or "xception"')
        parser.add_argument('-b', '--batch-size', metavar='batch_size', type=int,
                            required=False, nargs=1, default=[16])
        args = parser.parse_args()

        data_dir = args.data_dir[0]
        other_class = args.other_class[0].lower()
        weights_path = args.weights[0]
        mtype = args.mtype[0]
        batch_size = args.batch_size[0]

        # Validate arguments.
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)
        if not weights_path is None and not os.path.isfile(weights_path):
            print('"{}" is not a file'.format(weights_path), file=stderr)
            exit(2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)

        main(data_dir, other_class, weights_path, mtype, batch_size=batch_size)

    except KeyboardInterrupt:
        print('Program terminated prematurely')
