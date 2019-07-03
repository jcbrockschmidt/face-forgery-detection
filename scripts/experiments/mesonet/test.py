#!/usr/bin/env python3

"""
Tests a Meso4 model against a testing set.

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

from MesoNet.classifiers import Meso4

# Silence Tensorflow warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

BATCH_SIZE = 32


def main(data_dir, other_class, weights_path):
    """
    Tests a Meso4 model.

    Args:
        data_dir: Directory containing test classes, including "real"
            and `other_class`.
        other_class: Class other than "real" to test on.
        weights_path: Path to HDF5 weights file for the model.
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

    # Create data generators.
    test_data_generator = ImageDataGenerator(rescale=1/255)
    test_generator = test_data_generator.flow_from_directory(
        data_dir,
        classes=[other_class, 'real'],
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training')

    # Create model.
    model = Meso4()
    model.load(weights_path)

    # Test model.
    loss, acc = model.model.evaluate_generator(generator=test_generator,
                                               steps=len(test_generator))
    print('loss: {}'.format(loss))
    print('accuracy: {}'.format(acc))


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Trains a Meso4 model')
        parser.add_argument('data_dir', type=str, nargs=1,
                            help='directory containing test classes')
        parser.add_argument('other_class', metavar='class', type=str, nargs=1,
                            help='class other than "real" to test on')
        parser.add_argument('weights', type=str, nargs=1,
                            help='HDF5 weight file to initialize model with')
        args = parser.parse_args()

        # Validate arguments.
        data_dir = args.data_dir[0]
        other_class = args.other_class[0].lower()
        weights_path = args.weights[0]
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

        main(data_dir, other_class, weights_path)

    except KeyboardInterrupt:
        print('Program terminated prematurely')
