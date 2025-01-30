import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Reshape, Dense, Layer, Flatten
from src.machine_learning.mplc_error import MPLCLoss


class ModulusActivation(Layer):
    def __init__(self, modulus):
        super(ModulusActivation, self).__init__()
        self.modulus = modulus

    def call(self, inputs):
        return tf.math.floormod(inputs, self.modulus)
    
    def get_config(self):
        config = super(ModulusActivation, self).get_config()
        config.update({"modulus": self.modulus})
        return config
    

class ResizeLayer(Layer):
    def __init__(self, target_height, target_width, method='bilinear', **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_height = target_height
        self.target_width = target_width
        self.method = method

    def call(self, inputs):
        return tf.image.resize(inputs, [self.target_height, self.target_width], method=self.method)

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({
            'target_height': self.target_height,
            'target_width': self.target_width,
            'method': self.method
        })
        return config


class TransposeLayer(Layer):
    def __init__(self, perm, **kwargs):
        super(TransposeLayer, self).__init__(**kwargs)
        self.perm = perm

    def call(self, inputs):
        # Transpose the dimensions from (batch, n, m, k) to (batch, k, n, m)
        return tf.transpose(inputs, self.perm)

    def get_config(self):
        config = super(TransposeLayer, self).get_config()
        config.update({'perm': self.perm})
        return config
    

def create_neural_net(k=4, n=20, m=20, phase_restriction=2*np.pi/3, method=1, **kwargs):
    """
    Creating a neural net that performs phase accumulation for an
    (n x m) matrix input and (k x n x m) output.
    """

    print(f'{phase_restriction / np.pi}' + r'$\pi$')

    # Create a sequential model
    if method == 1:
        model = Sequential([
            # Convolutional base
            Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(n, m, 2)),
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

            # Reshape the output to use in a dense layer, assuming we flatten it and then expand again
            Flatten(),  # Flatten the output of the conv layers
            Dense(64, activation='relu'),  # Dense layer that processes the features
            Dense(n * m * k, activation='relu'), 
            Reshape((n, m, k)),

            # Additional convolutional layers to expand to k channels
            # Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
            # Conv2D(k, kernel_size=(1, 1), activation='sigmoid', padding='same'),  # k output channels
            TransposeLayer(perm=[0, 3, 1, 2]),
            ModulusActivation(modulus=phase_restriction) 
        ])

    else:
        model = Sequential([
            # Convolutional base
            Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(n, m, 2)),
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),

            # Reshape the output to use in a dense layer, assuming we flatten it and then expand again
            Flatten(),  # Flatten the output of the conv layers
            Dense(64, activation='relu'),  # Dense layer that processes the features
            Dense(128, activation='relu'),  # Dense layer that processes the features
            Reshape((16, 8, 1)),  # Reshape back to a spatial format
            ResizeLayer(n, m),

            # Additional convolutional layers to expand to k channels
            # Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'),
            Conv2D(k, kernel_size=(1, 1), activation='sigmoid', padding='same'),  # k output channels
            TransposeLayer(perm=[0, 3, 1, 2]),
            ModulusActivation(modulus=phase_restriction) 
        ])

    # Return the model
    return model
