import sys
import os
from pathvalidate import ValidationError, validate_filename

import numpy  as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

from src.machine_learning.nn import ModulusActivation, ResizeLayer, TransposeLayer
from src.machine_learning.generate_data import generate_data


def train_pnn(model, input_field=None, output_field=None, batch_size=32,
              steps_per_epoch=100, epochs=10, model_path=None, **kwargs):
    
    # Check for missing inputs
    if model_path is None:
        raise SyntaxError('Must input a model_path')
    
    # Compile path into a valid name
    current_directory = os.getcwd()
    full_path = os.path.normpath(os.path.join(current_directory, model_path))

    # Ensure the directory exists
    absolute_path = os.path.abspath(full_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    
    # # Tile the input and output data
    # n, m = input_field.shape
    # input_tiled = np.broadcast_to(input_field, (batch_size, n, m))
    # output_tiled = np.broadcast_to(output_field, (batch_size, n, m))
    
    # # Train the model on a single input, output pair
    # train_dataset = (input_tiled, output_tiled)
    # model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs)  

    # Perform the training on the input_field-output_field pair
    train_dataset = tf.data.Dataset.from_tensors((input_field, output_field))
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(batch_size)
    history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # Save the trained model
    model.save_weights(model_path)
    return history


def train_model(model, steps_per_epoch=100, epochs=30, batch_size=32, 
                model_path=None, input_field=None, h=None, **kwargs):
    """
    Train a neural network model.

    Parameters:
    - model: A Keras model instance
    - train_data: Input data for training.
    - train_labels: Labels for the training data.
    - epochs: Number of epochs to train the model.
    - batch_size: Size of batches for the training process.

    Returns:
    - Trained model.
    """
    
    # Ensure there is an appropriate path for saving the model
    if model_path is None:
        i = 0
        while True:
            i += 1
            model_name = input("Name trained model: ")
            model_path = f'/resources/neural_nets/{model_name}'

    # Extract physical parameters from the model
    _, k, n, m = model.output_shape
    phase_restriction = model.layers[-1].modulus
    
    # Compile path into a valid name
    current_directory = os.getcwd()
    full_path = os.path.normpath(os.path.join(current_directory, model_path))

    # Ensure the directory exists
    absolute_path = os.path.abspath(full_path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

    # Train model using generator
    train_gen =  generate_data(k, n, m, phase_restriction, input_field, h, batch_size=batch_size)

    # Create an explicit train dataset
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_signature=(
            tf.TensorSpec(shape=(None, 20, 20, 2), dtype=tf.float32),  # Input shape
            tf.TensorSpec(shape=(None, 4, 20, 20), dtype=tf.float32)   # Output shape
        )
    )

    # # DEBUG
    # x_batch, y_batch = next(train_gen)
    # print(f"Before training - x_batch shape: {x_batch.shape}, dtype: {x_batch.dtype}")  # Expected: (2, 20, 20, 2)
    # print(f"Before training - y_batch shape: {y_batch.shape}, dtype: {y_batch.dtype}")  # Expected: (2, 4, 20, 20)

    # Define steps_per_epoch based on desired training size
    model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=epochs)  

    # Save the model weights after training
    model.save_weights(model_path)


    # - - - - - - - - -  ARCHIVED CODE - - - - - - - - - #

    # # Check if the input is a string path to a model
    # if isinstance(model, str):

    #     # Load the model from the specified path
    #     model_path = model
    #     model = load_model(model,custom_objects = {
    #             'ModulusActivation': ModulusActivation,
    #             'TransposeLayer': TransposeLayer,
    #             'ResizeLayer': ResizeLayer
    #         })

    # elif isinstance(model, Model):
    #     # Use the model directly
    #     model = model

    # else:
    #     raise ValueError("model must be a Keras model instance or a path to a saved Keras model.")
