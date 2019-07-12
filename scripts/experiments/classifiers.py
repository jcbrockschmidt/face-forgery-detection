import keras.backend as K
from keras.applications import Xception as KerasXception
from keras.initializers import glorot_uniform, zeros
from keras.layers import BatchNormalization, Concatenate, Conv2D, Dense, Dropout, Flatten, Input, LeakyReLU, MaxPooling2D
from keras.models import Model as Model
from keras.optimizers import Adam

CLASS_MODES = {'binary', 'categorical'}
IMG_SIZE = 256

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
        self.model.load_weights(path)

    def save(self, path):
        """
        Save weights for a model.

        Args:
            path: Path to weights file.
        raise NotImplementedError()
        """
        self.model.load_weights(path)

    def fit_with_generator(self, generator, steps_per_epoch,
                           validation_data=None,
                           validation_steps=None,
                           class_weight=None,
                           epochs=1,
                           initial_epoch=0,
                           shuffle=True,
                           callbacks=None):
        """
        See keras.engine.training.Model.fit_generator.
        """
        return self.model.fit_generator(generator, steps_per_epoch,
                                        validation_data=validation_data,
                                        validation_steps=validation_steps,
                                        class_weight=class_weight,
                                        epochs=epochs,
                                        initial_epoch=initial_epoch,
                                        shuffle=shuffle,
                                        callbacks=callbacks)

    def evaluate_with_generator(self, generator):
        """
        Evaluate the model with a data generator.

        Args:
            generator: The data generator with samples to evaluate.

        Returns:
            Performance metrics.
        """
        return self.model.evaluate_generator(generator=generator,
                                             steps=len(generator),
                                             verbose=1)

    def set_metrics(self, metrics):
        """
        Sets the metrics used to judge the performance of the model.

        Args:
            metrics: Keras-compatible metrics.
        """
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=metrics)

class Meso1(Classifier):
    """
    Meso-1 model, borrowed from https://github.com/DariusAf/MesoNet
    """

    def __init__(self, learning_rate=0.001, dl_rate=1):
        self.model = self._init_model(dl_rate)
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def _init_model(self, dl_rate):
        x = Input(shape = (IMG_SIZE, IMG_SIZE, 3))

        x1 = Conv2D(16, (3, 3), dilation_rate = dl_rate, strides = 1, padding='same', activation = 'relu')(x)
        x1 = Conv2D(4, (1, 1), padding='same', activation = 'relu')(x1)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(8, 8), padding='same')(x1)

        y = Flatten()(x1)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)

class Meso4(Classifier):
    """
    Meso-4 model, borrowed from https://github.com/DariusAf/MesoNet
    """

    def __init__(self, learning_rate=0.001):
        self.model = self._init_model()
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def _init_model(self):
        x = Input(shape = (IMG_SIZE, IMG_SIZE, 3))

        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)

class MesoInception4(Classifier):
    """
    MesoInception-4 model, borrowed from https://github.com/DariusAf/MesoNet
    """

    def __init__(self, learning_rate=0.001):
        self.model = self._init_model()
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=self.optimizer,
                                 loss='mean_squared_error',
                                 metrics=['accuracy'])

    def InceptionLayer(self, a, b, c, d):
        """
        Creates an inception layer.

        Returns:
            Keras-compatible inception layer.
        """
        def func(x):
            x1 = Conv2D(a, (1, 1), padding='same', activation='relu')(x)

            x2 = Conv2D(b, (1, 1), padding='same', activation='relu')(x)
            x2 = Conv2D(b, (3, 3), padding='same', activation='relu')(x2)

            x3 = Conv2D(c, (1, 1), padding='same', activation='relu')(x)
            x3 = Conv2D(c, (3, 3), dilation_rate = 2, strides = 1, padding='same', activation='relu')(x3)

            x4 = Conv2D(d, (1, 1), padding='same', activation='relu')(x)
            x4 = Conv2D(d, (3, 3), dilation_rate = 3, strides = 1, padding='same', activation='relu')(x4)

            y = Concatenate(axis = -1)([x1, x2, x3, x4])

            return y
        return func

    def _init_model(self):
        x = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        x1 = self.InceptionLayer(1, 4, 4, 2)(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = self.InceptionLayer(2, 4, 4, 2)(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)

class MesoInc4Frozen16(MesoInception4):
    """
    A MesoInception-4 model where the convolutional layers are untrainable.
    Keeps the original classification layers.
    """

    FREEZE_BOUND = 27

    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        self.model = self._init_model()

        # Freeze the convolutional layers.
        for layer in self.model.layers[:self.FREEZE_BOUND]:
            layer.trainable = False
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

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
        self.model = KerasXception(weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                   include_top=False, pooling='avg')
        y = Dense(1, activation='sigmoid', name='prediction')(
            self.model.layers[-1].output)
        x = self.model.input
        self.model = Model(inputs=x, outputs=y)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

MODEL_MAP = {
    'meso1': Meso1,
    'meso4': Meso4,
    'mesoinception4': MesoInception4,
    'mesoinc4frozen16': MesoInc4Frozen16,
    'mesoinc4frozen48': MesoInc4Frozen48,
    'xception': Xception
}
