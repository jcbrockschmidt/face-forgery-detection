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
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sys import stderr

from classifiers import MODEL_MAP

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

def main(data_dir, save_dir, other_class, mtype='meso4', weights_path=None,
         epoch=1, transfer=False, batch_size=16):
    """
    Trains a model.

    Args:
        data_dir: Directory containing a "train" and "val" directory,
            each with a directory for the "real" and `other_class` classes.
        save_dir: Directory to save checkpoints and CSV file with loss and accuracy.
        mtype: Model type.  Should be "meso1", "meso4", "mesoinception4", or "mesoinc4frozen16"
        other_class: Other class to train on, the other being "real".
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
    train_real_dir = os.path.join(train_dir, 'real')
    other_class_dir = os.path.join(train_dir, other_class)
    valid_real_dir = os.path.join(valid_dir, 'real')
    valid_class_dir = os.path.join(valid_dir, other_class)
    if not os.path.exists(train_real_dir):
        print('ERROR: "{}" has no class "real"'.format(train_real_dir),
              file=stderr)
        exit(2)
    if not os.path.exists(other_class_dir):
        print('ERROR: "{}" has no class "{}"'.format(train_real_dir, other_class),
              file=stderr)
        exit(2)
    if not os.path.exists(valid_real_dir):
        print('ERROR: "{}" has no class "real"'.format(valid_real_dir),
              file=stderr)
        exit(2)
    if not os.path.exists(valid_class_dir):
        print('ERROR: "{}" has no class "{}"'.format(valid_real_dir, other_class),
              file=stderr)
        exit(2)

    # Make sure model is valid.
    if not mtype in MODEL_MAP:
        print('ERROR: "{}" is not a valid model type'.format(mtype),
              file=stderr)
        exit(2)

    # Create save directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create data generators.
    print('\nLoading training data from "{}"...'.format(train_dir))
    train_data_generator = ImageDataGenerator(rescale=1/255)
    train_generator = train_data_generator.flow_from_directory(
        train_dir,
        classes=[other_class, 'real'],
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    print('\nLoading validation data from "{}"...'.format(valid_dir))
    valid_data_generator = ImageDataGenerator(rescale=1/255)
    valid_generator = valid_data_generator.flow_from_directory(
        valid_dir,
        classes=[other_class, 'real'],
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    # Create model.
    model = MODEL_MAP[mtype]()
    if transfer:
        model.load_transfer(weights_path)
    elif not weights_path is None:
        model.load(weights_path)

    # Train model
    print('\nTraining {} model on class {}...\n'.format(mtype.upper(), other_class.upper()))
    callback = CustomCallback(save_dir, save_epoch=SAVE_EPOCH)
    model.fit_with_generator(
        train_generator, len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
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
            description='Trains a MesoNet model')
        parser.add_argument('data_dir', type=str, nargs=1,
                            help='directory containing a "train" and "val" directory')
        parser.add_argument('save_dir', type=str, nargs=1,
                            help='directory to save checkpoints and other data to')
        parser.add_argument('other_class', metavar='class', type=str, nargs=1,
                            help='class other than "real" to train on')
        parser.add_argument('-m', '--mtype', type=str, required=False, nargs=1,
                            default=['mesoinception4'],
                            help='model type, either {}'.format(models_str))
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
        other_class = args.other_class[0].lower()
        mtype = args.mtype[0].lower()
        weights_path = args.weights[0]
        epoch = args.epoch[0]
        batch_size = args.batch_size[0]
        gpu_frac = args.gpu_fraction[0]

        # Validate arguments.
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)
        if not weights_path is None and not os.path.isfile(weights_path):
            print('"{}" is not a file'.format(weights_path), file=stderr)
            exit(2)
        elif epoch < 0:
            print('epoch must be 0 or greater', file=stderr)
            exit(2)

        transfer = args.transfer
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

        main(data_dir, save_dir, other_class,
             mtype=mtype, weights_path=weights_path,
             epoch=epoch, transfer=transfer,
             batch_size=batch_size)

    except KeyboardInterrupt:
        print('Program terminated prematurely')
