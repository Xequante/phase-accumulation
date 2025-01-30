"""
Custom Error function for the Phase Accumulation ML model
"""


import numpy as np
from numpy import abs, sum
from src.propagation.propagate import mplc_propagate, free_space_propagate

import tensorflow as tf
from tensorflow.keras.losses import Loss


class MPLCLoss(Loss):
    def __init__(self, input_field, h, name="mplc_loss"):
        super().__init__(name=name)
        self.input_field = input_field
        self.h = h

    def call(self, y_true, y_pred):
        print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
        return mplc_error(y_true, y_pred, input_field=self.input_field, h=self.h)
    

def mplc_error(masks_true, masks_predicted, input_field, h):

    # Perform the MPLC propagation to calculate the true desired output
    print(masks_true.dtype)
    wavefront = input_field
    for i in range(masks_true.shape[1]):
        wavefront = mplc_propagate(wavefront, masks_true[:, i], h=h)
    output_true = tf.identity(wavefront)

    # Perform the MPLC propagation to calculate the predicted desired output
    wavefront = input_field
    for i in range(masks_predicted.shape[1]):
        wavefront = mplc_propagate(wavefront, masks_predicted[:, i], h=h)
    output_predicted = tf.identity(wavefront)

    # Calculate the optimal phase and amplitude offset
    c = tf.reduce_sum(tf.math.conj(output_predicted) * output_true)
    c /= tf.cast(tf.reduce_sum(tf.math.abs(output_predicted) ** 2), tf.complex64)

    # Check difference in the resulting wavefront and the desired output field
    return tf.reduce_sum(tf.math.abs(output_true - (c * output_predicted)) ** 2)
