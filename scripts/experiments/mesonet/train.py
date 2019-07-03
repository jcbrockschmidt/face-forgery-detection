#!/usr/bin/env python3

"""
Trains a Meso4 model.

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

from MesoNet.classifiers import Meso4

# Silence Tensorflow warnings.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

EPOCHS = 1000
BATCH_SIZE = 32
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
                  'Saving weights to "{}"'.format(self.best_path, self._best_acc))
            self.model.save_weights(self.best_path)

        # Save model weights.
        if epoch % self.save_epoch == 0:
            check_path = os.path.join(self.save_dir, '{}.hdf5'.format(epoch))
            print('Saving weights to "{}"...'.format(check_path))
            self.model.save_weights(check_path)

def main(data_dir, save_dir, train_class, weights_path=None, epoch=1):
    """
    Trains a Meso4 model.

    Args:
        data_dir: Directory containing a "train" and "val" directory,
            each with a directory for the "real" and `train_class` classes.
        save_dir: Directory to save checkpoints and CSV file with loss and accuracy.
        train_class: Other class to train on, the other being "real".
        weights_path: Path to HDF5 weights file to load model with.
            A new model will be created if set to None.
        epoch: Epoch to start on.
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
    train_class_dir = os.path.join(train_dir, train_class)
    valid_real_dir = os.path.join(valid_dir, 'real')
    valid_class_dir = os.path.join(valid_dir, train_class)
    if not os.path.exists(train_real_dir):
        print('ERROR: "{}" has no class "real"'.format(train_real_dir),
              file=stderr)
        exit(2)
    if not os.path.exists(train_class_dir):
        print('ERROR: "{}" has no class "{}"'.format(train_real_dir, train_class),
              file=stderr)
        exit(2)
    if not os.path.exists(valid_real_dir):
        print('ERROR: "{}" has no class "real"'.format(valid_real_dir),
              file=stderr)
        exit(2)
    if not os.path.exists(valid_class_dir):
        print('ERROR: "{}" has no class "{}"'.format(valid_real_dir, train_class),
              file=stderr)
        exit(2)

    # Create save directory if it does not exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create data generators.
    train_data_generator = ImageDataGenerator(rescale=1/255)
    train_generator = train_data_generator.flow_from_directory(
        train_dir,
        classes=[train_class, 'real'],
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training')

    valid_data_generator = ImageDataGenerator(rescale=1/255)
    valid_generator = valid_data_generator.flow_from_directory(
        valid_dir,
        classes=[train_class, 'real'],
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training')

    # Create model.
    model = Meso4()
    if not weights_path is None:
        model.load(weights_path)

    # Train model
    callback = CustomCallback(save_dir, save_epoch=SAVE_EPOCH)
    model.model.fit_generator(
        train_generator, len(train_generator),
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        epochs=EPOCHS,
        initial_epoch=epoch,
        verbose=1,
        shuffle=True,
        callbacks=[callback])


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(
            description='Trains a Meso4 model')
        parser.add_argument('data_dir', type=str, nargs=1,
                            help='directory containing a "train" and "val" directory')
        parser.add_argument('save_dir', type=str, nargs=1,
                            help='directory to save checkpoints and other data to')
        parser.add_argument('train_class', metavar='class', type=str, nargs=1,
                            help='class other than "real" to train on')
        parser.add_argument('-w', '--weights', type=str, required=False, nargs=1,
                            help='HDF5 weight file to initialize model with')
        parser.add_argument('-e', '--epoch', type=int, required=False, nargs=1,
                            help='epoch to start on')
        args = parser.parse_args()

        # Validate arguments.
        data_dir = args.data_dir[0]
        save_dir = args.save_dir[0]
        train_class = args.train_class[0].lower()
        if args.weights is None:
            weights_path = None
        else:
            weights_path = args.weights[0]
        if args.epoch is None:
            epoch = 0
        else:
            epoch = args.epoch[0]
        if not os.path.isdir(data_dir):
            print('"{}" is not a directory'.format(data_dir), file=stderr)
            exit(2)
        if not weights_path is None and not os.path.isfile(weights_path):
            print('"{}" is not a file'.format(weights_path), file=stderr)
            exit(2)
        elif epoch < 0:
            print('epoch must be 0 or greater', file=stderr)
            exit(2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)

        main(data_dir, save_dir, train_class, weights_path=weights_path, epoch=epoch)

    except KeyboardInterrupt:
        print('Program terminated prematurely')
