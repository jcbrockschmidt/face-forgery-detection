import keras.backend as K
from keras.applications import Xception as KerasXception
from keras.initializers import glorot_uniform, zeros
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
        pass

    def save(self, path):
        """
        Save weights for a model.

        Args:
            path: Path to weights file.
        """
        pass

    def fit_with_generator(self, generator, steps_per_epoch,
                           validation_data=None,
                           validation_steps=None,
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):
        """
        See keras.engine.training.Model.fit_generator.
        """
        pass

    def evaluate_with_generator(self, generator):
        """
        Evaluate the model with a data generator.

        Args:
            generator: The data generator with samples to evaluate.

        Returns:
            Loss and accuracy.
        """
        pass

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
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        self.model.model.fit_generator(generator, steps_per_epoch,
                                       validation_data=validation_data,
                                       validation_steps=validation_steps,
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
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        return self.model.model.fit_generator(generator, steps_per_epoch,
                                              validation_data=validation_data,
                                              validation_steps=validation_steps,
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
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        return self.model.model.fit_generator(generator, steps_per_epoch,
                                              validation_data=validation_data,
                                              validation_steps=validation_steps,
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
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        return self.model.model.fit_generator(generator, steps_per_epoch,
                                              validation_data=validation_data,
                                              validation_steps=validation_steps,
                                              epochs=epochs,
                                              initial_epoch=initial_epoch,
                                              shuffle=shuffle,
                                              callbacks=callbacks)

    def evaluate_with_generator(self, generator):
        return self.model.model.evaluate_generator(generator=generator,
                                              steps=len(generator),
                                              verbose=1)

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
        self.optimizer = Adam(lr = learning_rate)
        self.model = KerasXception(weights=None, input_shape=(256, 256, 3), classes=1)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def fit_with_generator(self, generator, steps_per_epoch,
                           validation_data=None,
                           validation_steps=None,
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):

        return self.model.fit_generator(generator, steps_per_epoch,
                                        validation_data=validation_data,
                                        validation_steps=validation_steps,
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
    'xception': Xception
}
