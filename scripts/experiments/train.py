#!/usr/bin/env python3

"""
Trains a model.

```
./train.py <DATA_DIR> <SAVE_DIR> <CLASS> -w <WEIGHT_PATH> -e <EPOCH>
```

`DATA_DIR` should be created by `scripts/data_prep/make_video_splits.py`.


Example of training a new model:

```
./train.py data/c0 models/my-model x2f
```


Starting that same model on epoch 50:

```
./train.py data/c0 models/my-model x2f -w models/my-model/50.hdf5 -e 50
```
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show Tensorflow errors.

import argparse
import csv
from keras.backend.tensorflow_backend import set_session
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sys import stderr

from classifiers import CLASS_MODES, MODEL_MAP
from utils import create_data_generator

# Silence Tensorflow warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

EPOCHS = 1000
SAVE_EPOCH = 5

class CustomCallback(Callback):
    """
    Saves model loss and accuracy after every epoch and occassionally
    creates checkpoints

    Attributes:
        best_path: Path to checkpoint for model with the best accuracy.
        data_path: Path to CSV file that stores loss and accuracy for training
            and validation set for each epoch.
        save_dir: Directory to save data and checkpoints to.
    """

    def __init__(self, save_dir, save_epoch=10):
        """
        Args:
            save_dir: Directory to save data and checkpoints to.
            save_epoch: Epochs between each checkpoint save.
        """
        super().__init__()
        self.save_dir = save_dir
        self.best_path = os.path.join(self.save_dir, 'best.hdf5')
        self.data_path = os.path.join(self.save_dir, 'data.csv')
        self.save_epoch = save_epoch

        # TODO: Load current best model (if it exists) to get current
        #       best accuracy.
        self._best_acc = 0

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if not os.path.exists(self.data_path):
            # Write column labels.
            with open(self.data_path, 'a') as f:
                cw = csv.writer(f)
                labels = ['epoch', 'val_loss', 'val_acc', 'loss', 'acc']
                cw.writerow(labels)

    def on_epoch_end(self, epoch, logs={}):
        # Write loss and accuracy data to disk.
        val_acc = logs['val_acc']
        with open(self.data_path, 'a') as f:
            cw = csv.writer(f)
            row = [
                epoch,
                logs['val_loss'],
                val_acc,
                logs['loss'],
                logs['acc']
            ]
            cw.writerow(row)

        # Check if validation accuracy exceeds previous best.
        if val_acc > self._best_acc:
            self._best_acc = val_acc
            print('New best found with accuracy {}.  ' \
                  'Saving weights to "{}"'.format(self._best_acc, self.best_path))
            self.model.save_weights(self.best_path)

        # Save model weights.
        if epoch % self.save_epoch == 0:
            check_path = os.path.join(self.save_dir, '{}.hdf5'.format(epoch))
            print('Saving weights to "{}"...'.format(check_path))
            self.model.save_weights(check_path)

def main(data_dir, save_dir, other_classes, mtype,
         class_mode='binary',
         weights_path=None,
         epoch=1,
         transfer=False,
         batch_size=16):
    """
    Trains a model.

    Args:
        data_dir: Directory containing a "train" and "val" directory,
            each with a directory for the "real" and `other_class` classes.
        save_dir: Directory to save checkpoints and CSV file with loss and accuracy.
        other_classes: Other classes to train on (wherein the default class is "real").
        mtype: Model type.  Should be "meso1", "meso4", "mesoinception4", or "mesoinc4frozen16"
        class_mode: See `keras.preprocessing.image.ImageDataGenerator.flow_from_directory`.
        weights_path: Path to HDF5 weights file to load model with.
            A new model will be created if set to None.
        epoch: Epoch to start on.
        transfer: Whether to transfer from a MesoInception4 to a MesoInc4Frozen4.
            mtype should be either "mesoinception4" or "mesoinc4frozen16",
            and a weights_path should be specified.
        batch_size: Number of images to process at a time.
    """
    # Make sure training and validation set exists.
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'val')
    if not os.path.exists(train_dir):
        print('ERROR: "{}" has no "train" set'.format(data_dir),
              file=stderr)
        exit(2)
    if not os.path.exists(train_dir):
        print('ERROR: "{}" has no "val" set'.format(data_dir),
              file=stderr)
        exit(2)

    # Make sure classes exist.
    for c in other_classes + ['real']:
        class_train_dir = os.path.join(train_dir, c)
        class_valid_dir = os.path.join(valid_dir, c)
        if not os.path.exists(class_train_dir):
            print('ERROR: "{}" has no class "{}"'.format(train_dir, c),
                  file=stderr)
            exit(2)
            if not os.path.exists(class_valid_dir):
                print('ERROR: "{}" has no class "{}"'.format(valid_dir, c),
                      file=stderr)
            exit(2)

    # Make sure model is valid.
    if not mtype in MODEL_MAP:
        print('ERROR: "{}" is not a valid model type'.format(mtype),
              file=stderr)
        exit(2)

    # Make sure classification mode is valid.
    if not class_mode in CLASS_MODES:
        print('ERROR: "{}" is not a valid classification mode'.format(class_mode),
              file=stderr)
        exit(2)

    # Create save directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create data generators.
    print('\nLoading training data from "{}"...'.format(train_dir))
    train_generator, class_weight = create_data_generator(
        train_dir, other_classes, batch_size, class_mode)

    print('\nLoading validation data from "{}"...'.format(valid_dir))
    valid_generator, _ = create_data_generator(
        valid_dir, other_classes, batch_size, class_mode)

    # Create model.
    model = MODEL_MAP[mtype]()
    if transfer:
        print('\nTransferring MESOINCEPTION4 model from "{}"'.format(weights_path))
        model.load_transfer(weights_path)
    elif not weights_path is None:
        print('\nLoading {} model from "{}"'.format(mtype.upper(), weights_path))
        model.load(weights_path)

    # Train model.
    classes_str = ', '.join(other_classes)
    print('\nTraining {} model as a {} classifier on classes {}...\n'.format(
        mtype.upper(), class_mode, classes_str.upper()))
    callback = CustomCallback(save_dir, save_epoch=SAVE_EPOCH)
    model.fit_with_generator(
        train_generator, len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        class_weight=class_weight,
        epochs=EPOCHS,
        initial_epoch=epoch,
        shuffle=True,
        callbacks=[callback])


if __name__ == '__main__':
    try:
        # Construct a string listing all model types in the format
        #
        #    'model1, model2, model3, or model4'
        #
        models = ['"{}"'.format(k.lower()) for k in MODEL_MAP.keys()]
        models_str = '{}, or {}'.format(', '.join(models[:-1]), models[-1])

        parser = argparse.ArgumentParser(
            description='Trains a model')
        parser.add_argument('-d', '--data-dir', dest='data_dir', type=str,
                            required=True, nargs=1,
                            help='directory containing a "train" and "val" directory')
        parser.add_argument('-s', '--save-dir', dest='save_dir', type=str,
                            required=True, nargs=1,
                            help='directory to save checkpoints and other data to')
        parser.add_argument('-c', '--classes', metavar='class', dest='other_classes',
                            type=str, nargs='+', required=True,
                            help='classes other than "real" to train on')
        parser.add_argument('-m', '--mtype', type=str, required=True, nargs=1,
                            help='model type, either {}'.format(models_str))
        parser.add_argument('-cm', '--class-mode', dest='class_mode', type=str,
                            required=False, nargs=1, default=['binary'],
                            help='type of "binary" or "categorical"')
        parser.add_argument('-w', '--weights', type=str, required=False, nargs=1,
                            default=[None],
                            help='HDF5 weight file to initialize model with')
        parser.add_argument('-e', '--epoch', type=int, required=False, nargs=1,
                            default=[0],
                            help='epoch to start on')
        parser.add_argument('-t', '--transfer',
                            action='store_const', const=True, default=False,
                            help='transfer a mesoinception4 to a mesoinc4frozen16')
        parser.add_argument('-b', '--batch-size', metavar='batch_size', type=int,
                            required=False, nargs=1, default=[16],
                            help='number of images to read at a time')
        parser.add_argument('-g', '--gpu-fraction', metavar='batch_size', type=float,
                            required=False, nargs=1, default=[1.0],
                            help='maximum fraction of the GPU\'s memory the ' \
                            'model is allowed to use, between 0.0 and 1.0')
        args = parser.parse_args()

        data_dir = args.data_dir[0]
        save_dir = args.save_dir[0]
        other_classes = args.other_classes
        mtype = args.mtype[0].lower()
        class_mode = args.class_mode[0].lower()
        weights_path = args.weights[0]
        epoch = args.epoch[0]
        transfer = args.transfer
        batch_size = args.batch_size[0]
        gpu_frac = args.gpu_fraction[0]

        # Validate arguments.
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)
        if not weights_path is None and not os.path.isfile(weights_path):
            print('"{}" is not a file'.format(weights_path), file=stderr)
            exit(2)
        if epoch < 0:
            print('epoch must be 0 or greater', file=stderr)
            exit(2)
        if transfer:
            if not mtype in ('mesoinc4frozen16', 'mesoinc4frozen48'):
                print('Can only transfer to a "mesoinc4frozen16" or "mesoinc4frozen48" model',
                      file=stderr)
                exit(2)
            if weights_path is None:
                print('Please specify a weights_path for transferring',
                      file=stderr)
                exit(2)

        if gpu_frac < 0 or gpu_frac > 1:
            print('gpu-fraction must be between 0.0 and 1.0', file=stderr)
            exit(2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
        sess = tf.Session(config=config)
        set_session(sess)

        main(data_dir, save_dir, other_classes, mtype,
             class_mode=class_mode,
             weights_path=weights_path,
             epoch=epoch,
             transfer=transfer,
             batch_size=batch_size)

    except KeyboardInterrupt:
        print('Program terminated prematurely')
