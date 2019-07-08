import keras.backend as K
from keras.initializers import glorot_uniform, zeros
from keras.optimizers import Adam
from MesoNet.classifiers import MesoInception4

class MesoInc4Frozen16(MesoInception4):
    """
    A MesoInception-4 model where the convolutional layers are untrainable.
    Keeps the original classification layers.
    """

    FREEZE_BOUND = 27  # TODO

    def __init__(self, learning_rate=0.001):
        super().__init__(learning_rate=learning_rate)
        # Freeze the convolutional layers.
        for layer in self.model.layers[:self.FREEZE_BOUND]:
            layer.trainable = False
        self.optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=self.optimizer,
                           loss='mean_squared_error',
                           metrics=['accuracy'])

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
