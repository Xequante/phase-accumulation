import sys
from pathvalidate import ValidationError, validate_filename

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import Model

from src.machine_learning.nn import ModulusActivation
from src.machine_learning.generate_data import generate_data


def train_model(model, epochs=10, batch_size=32, model_path=None):
    """
    Train a neural network model.

    Parameters:
    - model: either a Keras model instance or a path to a saved Keras model.
    - train_data: Input data for training.
    - train_labels: Labels for the training data.
    - epochs: Number of epochs to train the model.
    - batch_size: Size of batches for the training process.

    Returns:
    - Trained model.
    """

    # Extract physical parameters from the model
    _, k, n, m, = model.output_shape
    phase_restriction = model.layers[-1].modulus

    # Check if the input is a string path to a model
    if isinstance(model, str):
        # Load the model from the specified path
        model_path = model
        model = load_model(model, custom_objects={'ModulusActivation': ModulusActivation})

    elif isinstance(model, Model):
        # Use the model directly
        model = model

    else:
        raise ValueError("model must be a Keras model instance or a path to a saved Keras model.")
    
    # Ensure there is an appropriate path for saving the model
    if model_path is None:
        i = 0
        while True:
            i += 1
            model_name = input("Name trained model: ")
            model_path = f'/resources/neural_nets/{model_name}'
            try:
                validate_filename(model_path)
            except ValidationError as e:
                print("{}\n".format(e), file=sys.stderr)
                if i == 3:
                    raise ValidationError('Issue with finding appropriate file name')
            else:
                break
    else:
        try:
            validate_filename(model_path)
        except ValidationError as e:
            print("{}\n".format(e), file=sys.stderr)
            raise ValidationError('Issue with file path')

    # Compile the model if not already compiled
    if not model.optimizer:
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model using generator
    train_gen =  generate_data(k, n, m, phase_restriction, batch_size=batch_size)

    # Define steps_per_epoch based on desired training size
    model.fit(train_gen, steps_per_epoch=100, epochs=epochs)  

    # Save the model after training
    model.save(model_path)
 