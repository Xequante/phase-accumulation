import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose, Reshape, Dense, Layer


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
    

def create_neural_net(k=4, n=50, m=60, phase_restriction=2*np.pi/3):
    """
    Creating a neural net that performs phase accumulation for an
    (n x m) matrix input and (k x n x m) output.
    """

    # Create a sequential model
    model = Sequential([
        # Convolutional base
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(n, m, 1)),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),

        # Reshape the output to use in a dense layer, assuming we flatten it and then expand again
        Reshape((-1,)),  # Flatten the output of the conv layers
        Dense(64 * n * m, activation='relu'),  # Dense layer that processes the features
        Reshape((n, m, 64)),  # Reshape back to a spatial format

        # Additional convolutional layers to expand to k channels
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        Conv2D(k, kernel_size=(1, 1), activation='sigmoid', padding='same'),  # k output channels
        ModulusActivation(modulus=phase_restriction) 
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # # Train the model
    # model.fit(x_train, y_train, epochs=10, batch_size=32)

    # # Evaluate the model
    # loss, accuracy = model.evaluate(x_test, y_test)
    # print('Accuracy:', accuracy)