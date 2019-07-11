import keras.backend as K
from keras.applications import Xception as KerasXception
from keras.initializers import glorot_uniform, zeros
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.models import Model as Model
from keras.optimizers import Adam
import MesoNet.classifiers as mesonet_classifiers

class Classifier:
    """
    Interface for face classifiers.
    """

    def load(self, path):
        """
        Load weights for a model.

        Args:
            path: Path to weights file.
        """
        raise NotImplementedError()

    def save(self, path):
        """
        Save weights for a model.

        Args:
            path: Path to weights file.
        raise NotImplementedError()
        """
        raise NotImplementedError()

    def fit_with_generator(self, generator, steps_per_epoch,
                           validation_data=None,
                           validation_steps=None,
                           class_weights=None,
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):
        """
        See keras.engine.training.Model.fit_generator.
        """
        raise NotImplementedError()

    def evaluate_with_generator(self, generator):
        """
        Evaluate the model with a data generator.

        Args:
            generator: The data generator with samples to evaluate.

        Returns:
            Loss and accuracy.
        """
        raise NotImplementedError()

class Meso1(Classifier):
    """
    Wrapper for a Meso-1 model.
    """

    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.model = mesonet_classifiers.Meso1(self.lr)

    def load(self, path):
        self.model.load(path)

    def save(self, path):
        self.model.load(path)

    def fit_with_generator(self, generator, steps_per_epoch,
                           validation_data=None,
                           validation_steps=None,
                           class_weight=None,
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        self.model.model.fit_generator(generator, steps_per_epoch,
                                       validation_data=validation_data,
                                       validation_steps=validation_steps,
                                       class_weight=class_weight,
                                       epochs=epochs,
                                       initial_epoch=initial_epoch,
                                       shuffle=shuffle,
                                       callbacks=callbacks)

    def evaluate_with_generator(self, generator):
        return self.model.model.evaluate_generator(generator=generator,
                                                   steps=len(generator),
                                                   verbose=1)

class Meso4(Classifier):
    """
    Wrapper for a Meso-4 model.
    """

    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.model = mesonet_classifiers.Meso4(self.lr)

    def load(self, path):
        self.model.load(path)

    def save(self, path):
        self.model.load(path)

    def fit_with_generator(self, generator, steps_per_epoch,
                           validation_data=None,
                           validation_steps=None,
                           class_weight=None,
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        return self.model.model.fit_generator(generator, steps_per_epoch,
                                              validation_data=validation_data,
                                              validation_steps=validation_steps,
                                              class_weight=class_weight,
                                              epochs=epochs,
                                              initial_epoch=initial_epoch,
                                              shuffle=shuffle,
                                              callbacks=callbacks)

    def evaluate_with_generator(self, generator):
        return self.model.model.evaluate_generator(generator=generator,
                                                   steps=len(generator),
                                                   verbose=1)

class MesoInception4(Classifier):
    """
    Wrapper for a MesoInception-4 model.
    """

    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.model = mesonet_classifiers.MesoInception4(learning_rate=self.lr)

    def load(self, path):
        self.model.load(path)

    def save(self, path):
        self.model.load(path)

    def fit_with_generator(self, generator, steps_per_epoch,
                           validation_data=None,
                           validation_steps=None,
                           class_weight=None,
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        return self.model.model.fit_generator(generator, steps_per_epoch,
                                              validation_data=validation_data,
                                              validation_steps=validation_steps,
                                              class_weight=class_weight,
                                              epochs=epochs,
                                              initial_epoch=initial_epoch,
                                              shuffle=shuffle,
                                              callbacks=callbacks)

    def evaluate_with_generator(self, generator):
        return self.model.model.evaluate_generator(generator=generator,
                                                   steps=len(generator),
                                                   verbose=1)

class MesoInc4Frozen16(Classifier):
    """
    A MesoInception-4 model where the convolutional layers are untrainable.
    Keeps the original classification layers.
    """

    FREEZE_BOUND = 27

    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.model = mesonet_classifiers.MesoInception4(learning_rate=self.lr).model

        # Freeze the convolutional layers.
        for layer in self.model.layers[:self.FREEZE_BOUND]:
            layer.trainable = False
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save_weights(path)

    def fit_with_generator(self, generator, steps_per_epoch,
                           validation_data=None,
                           validation_steps=None,
                           class_weight=None,
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        return self.model.fit_generator(generator, steps_per_epoch,
                                        validation_data=validation_data,
                                        validation_steps=validation_steps,
                                        class_weight=class_weight,
                                        epochs=epochs,
                                        initial_epoch=initial_epoch,
                                        shuffle=shuffle,
                                        callbacks=callbacks)

    def evaluate_with_generator(self, generator):
        return self.model.evaluate_generator(generator=generator,
                                             steps=len(generator),
                                             verbose=1)

    def load_transfer(self, path):
        """
        Loads weights from a MesoInception-4 model.

        Args:
            Path to weights file for a MesoInception-4 model.
        """
        self.model.load_weights(path)
        self.reset_classification()

    def reset_classification(self):
        """
        Reinitializes the weights for the classification layers.
        """
        for layer in self.model.layers[self.FREEZE_BOUND:]:
            old_weights = layer.get_weights()
            if len(old_weights) == 2:
                weights = glorot_uniform()(old_weights[0].shape).eval(session=K.get_session())
                bias = zeros()(old_weights[1].shape).eval(session=K.get_session())
                new_weights = [weights, bias]
                layer.set_weights(new_weights)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

class MesoInc4Frozen48(Classifier):
    """
    A MesoInception-4 model where the convolutional layers are untrainable
    and the flat/classifier layers are made up of 3 layers of 16 neurons.
    """

    FREEZE_BOUND = 27

    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.model = mesonet_classifiers.MesoInception4(learning_rate=self.lr).model

        # Freeze the convolutional layers.
        for layer in self.model.layers[:self.FREEZE_BOUND]:
            layer.trainable = False

        # Add the new flat layers, ignoring the original layers.
        y = self.model.layers[self.FREEZE_BOUND - 1].output
        y = Flatten()(y)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid', name='prediction')(y)

        # Recreate the model.
        x = self.model.input
        self.model = Model(inputs=x, outputs=y)
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save_weights(path)

    def fit_with_generator(self, generator, steps_per_epoch,
                           validation_data=None,
                           validation_steps=None,
                           class_weight=None,
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        return self.model.fit_generator(generator, steps_per_epoch,
                                        validation_data=validation_data,
                                        validation_steps=validation_steps,
                                        class_weight=class_weight,
                                        epochs=epochs,
                                        initial_epoch=initial_epoch,
                                        shuffle=shuffle,
                                        callbacks=callbacks)

    def evaluate_with_generator(self, generator):
        return self.model.evaluate_generator(generator=generator,
                                             steps=len(generator),
                                             verbose=1)

    def load_transfer(self, path):
        """
        Loads weights from a MesoInception-4 model.

        Args:
            Path to weights file for a MesoInception-4 model.
        """
        meso = MesoInception4()
        meso.load(path)
        for i, layer in enumerate(self.model.layers[:self.FREEZE_BOUND]):
            weights = layer.get_weights()
            self.model.layers[i].set_weights(weights)
        self.reset_classification()

    def reset_classification(self):
        """
        Reinitializes the weights for the classification layers.
        """
        for layer in self.model.layers[self.FREEZE_BOUND:]:
            old_weights = layer.get_weights()
            if len(old_weights) == 2:
                weights = glorot_uniform()(old_weights[0].shape).eval(session=K.get_session())
                bias = zeros()(old_weights[1].shape).eval(session=K.get_session())
                new_weights = [weights, bias]
                layer.set_weights(new_weights)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

class Xception(Classifier):
    """
    An Xception model for detecting faces.
    """

    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.optimizer = Adam(lr=learning_rate)
        self.model = KerasXception(weights=None, input_shape=(256, 256, 3),
                                   include_top=False, pooling='avg')
        y = Dense(1, activation='sigmoid', name='prediction')(
            self.model.layers[-1].output)
        x = self.model.input
        self.model = Model(inputs=x, outputs=y)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def load(self, path):
        """
        Load weights for a model.

        Args:
            path: Path to weights file.
        """
        self.model.load_weights(path)

    def save(self, path):
        """
        Save weights for a model.

        Args:
            path: Path to weights file.
        """
        self.model.save_weights(path)

    def fit_with_generator(self, generator, steps_per_epoch,
                           validation_data=None,
                           validation_steps=None,
                           class_weight=None,
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        return self.model.fit_generator(generator, steps_per_epoch,
                                        validation_data=validation_data,
                                        validation_steps=validation_steps,
                                        class_weight=class_weight,
                                        epochs=epochs,
                                        initial_epoch=initial_epoch,
                                        shuffle=shuffle,
                                        callbacks=callbacks)

    def evaluate_with_generator(self, generator):
        return self.model.evaluate_generator(generator=generator,
                                             steps=len(generator),
                                             verbose=1)

MODEL_MAP = {
    'meso1': Meso1,
    'meso4': Meso4,
    'mesoinception4': MesoInception4,
    'mesoinc4frozen16': MesoInc4Frozen16,
    'mesoinc4frozen48': MesoInc4Frozen48,
    'xception': Xception
}
