from src.machine_learning.mplc_error import PNNLoss
from src.machine_learning.nn import create_neural_net
from src.machine_learning.pnn import create_pnn
from src.machine_learning.compile import mplc_compile
from src.machine_learning.training import train_model, train_pnn
from src.utilities.parameters_dictionary import parameters_dictionary
from src.analysis.pnn_analysis import pnn_analysis
from src.analysis.loss_analysis import loss_analysis
from numpy import pi

import numpy as np
import tensorflow as tf


def main():

    # Generate a dictionary with necessary physical parameters
    parameters = parameters_dictionary(phase_restriction=pi, k=4, output_index=2)

    # Run loss analysis
    k_vals = [6, 10]
    loss_analysis(min_phase=pi/8, max_phase=3*pi / 4, num_models=20, 
                  load=True, only_load=False, k_vals=k_vals, 
                  extra_plots=False, only_plot=False,
                  label='LG', 
                  **parameters)

    # Define a path where the model will be saved
    # model_path = 'resources/neural_nets/pnn_model_1_mask.h5'

    # Create and train the model
    # create_pnn_model(model_path, **parameters)

    # pnn_extra_training(model_path, epochs=1, **parameters)
    # # check_pnn_model(model_path, **parameters)
    # # train_new_model(model_path, **parameters)


# Training a PINN (Physics Informed Neural Network)
def create_pnn_model(model_path, **kwargs):
    model = create_pnn(**kwargs)
    model.compile(optimizer='adam', 
                  loss=PNNLoss(),
                  metrics=['accuracy'])
    train_pnn(model, model_path=model_path, **kwargs)


# Extra training for the PNN Model
def pnn_extra_training(model_path, epochs=10, **kwargs):
    model = create_pnn(**kwargs)
    model.load_weights(model_path)
    model.compile(optimizer='adam', 
                  loss=PNNLoss(),
                  metrics=['accuracy'])
    train_pnn(model, model_path=model_path, epochs=epochs, **kwargs)
    weights = model.get_weights()

    # Perform MPLC propagation with these weights as the phase masks
    pnn_analysis(weights, model, **kwargs)


# Checking the success of the PINN
def check_pnn_model(model_path, **kwargs):
    model = create_pnn(**kwargs)
    model.load_weights(model_path)

    # Perform MPLC propagation with these weights as the phase masks
    pnn_analysis(model, **kwargs)


# Training a fresh neural network
def train_new_model(model_path, **kwargs):
    model = create_neural_net(**kwargs) 
    mplc_compile(model, **kwargs)
    train_model(model, model_path=model_path, **kwargs)


# Re-Training an existing neural network
def train_loaded_model(model_path, **kwargs):
    model = create_neural_net(**kwargs)
    mplc_compile(model, **kwargs)
    model.load_weights(model_path)
    train_model(model, model_path=model_path, **kwargs)

if __name__ == '__main__':
    main()
