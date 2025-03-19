import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Reshape, Dense, Layer, Flatten

from src.propagation.propagate import mplc_propagate


class MPLCLayer(Layer):

    def __init__(self, fsk, phase_restriction=2*np.pi, **kwargs):
        super(MPLCLayer, self).__init__(**kwargs)
        self.fsk = fsk
        self.pr = phase_restriction / 2

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[1:]),
                                 initializer='random_normal',
                                 trainable=True,
                                 name='kernel')
        
    def call(self, inputs):
        
        # Convert weights to a phase restricted mask
        phases = self.pr * tf.math.sin(self.w)

        # Perform MPLC propagation
        mask = tf.cast(phases, tf.float32)
        return mplc_propagate(inputs, mask, h=self.fsk)

    def get_config(self):
        config = super(MPLCLayer, self).get_config()
        config.update({'fsk': self.fsk})
        return config
    

def create_pnn(k=4, n=20, m=20, phase_restriction=2*np.pi/3, h=None, **kwargs):
    """
    Creating a neural net that performs phase accumulation for an
    (n x m) matrix input and (k x n x m) output.
    """

    # Print out phase restriction
    print('\n'*2)
    print('-'*36)
    print(f'PNN with Phase Restriction {phase_restriction / np.pi:.4f}π')
    print('-'*36)

    # Create a sequential model
    layer_list = [MPLCLayer(fsk=h, phase_restriction=phase_restriction, input_shape=(n, m))]
    for i in range(k-1):
        layer_list.append(MPLCLayer(fsk=h, phase_restriction=phase_restriction))
    model = Sequential(layer_list)

    # Return the model
    return model