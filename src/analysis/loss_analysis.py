from src.machine_learning.mplc_error import PNNLoss
from src.machine_learning.pnn import create_pnn
from src.machine_learning.training import train_pnn
from src.machine_learning.loss_threshold import HaltCallback
from src.analysis.pnn_analysis import pnn_analysis

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loss_analysis(epochs=40, steps_per_epoch=100, batch_size=32, 
                  loss_threshold=None, min_loss=-51200,
                  min_phase=np.pi / 2, max_phase=2*np.pi,
                  num_models:int = 16, k_vals=None, 
                  load: bool = False, only_load: bool = True, 
                  extra_plots=False, only_plot: bool = False,
                  label: str = None, repair = False,
                  **parameters):

    # Identify number of masks 
    if k_vals is None:
        k_vals = [parameters['k']]

    # Calculate the minimum loss for any batch
    min_batch_loss = min_loss * batch_size
    # TODO: set up something that stops the training once close enough to this min_batch_loss

    # Specify set of phase restrictions which will be tested
    phase_restrictions = np.linspace(min_phase, max_phase, num_models)
    phase_restrictions = np.concatenate((phase_restrictions, np.linspace(np.pi / 2, 2 * np.pi, 4)))
    phase_restrictions = np.unique(phase_restrictions)
    num_models = len(phase_restrictions)

    # Initialize vector for loss values
    loss_values = np.zeros((len(k_vals), num_models))

    # Create loss callback
    training_callbacks = []
    if loss_threshold is not None:
        training_callbacks.append(HaltCallback(loss_threshold))

    # Create a model for each phase_restriction
    for j, k in enumerate(k_vals):

        # Set k to the selected value
        parameters['k'] = k

        # Create masks
        for i in range(num_models):

            # Identify the phase restriction for this model
            pr = phase_restrictions[i]
            parameters['phase_restriction'] = pr

            # Define the path where the model will be saved
            pr_string = f'{pr:.4f}'.replace('.', '_')
            if label is None:
                model_path = f'resources/neural_nets/pnn_models/pnn_model_k_{k}_pr_{pr_string}.h5'
            else:
                model_path = f'resources/neural_nets/pnn_models/pnn_model_{label}_k_{k}_pr_{pr_string}.h5'

            # Initialize the model infrastructure
            model = create_pnn(**parameters)

            # Check if we are just generating plots
            if only_plot:
                if (k == 10 and i == 4):
                    # Load the weights and get a quick loss measurement
                    model.load_weights(model_path)
                    model.compile(optimizer='adam', 
                                loss=PNNLoss(),
                                metrics=['accuracy'])
                    pnn_analysis(model, **parameters)
                    return
                else:
                    continue

            # Chose between loading pre-existing models, and creating new ones
            elif load and os.path.exists(model_path):
                
                # Load the weights into the model
                model.load_weights(model_path)
                model.compile(optimizer='adam', 
                            loss=PNNLoss(),
                            metrics=['accuracy'])

                # Repairing weights <correcting former error>
                if repair:
                    weights = model.get_weights()
                    weights = [tf.transpose(w) for w in weights]
                    model.set_weights(weights)

                # Get a quick loss measurement
                history = train_pnn(model, model_path=model_path, batch_size=batch_size,
                                    epochs=1, steps_per_epoch=1, **parameters)
            
            # If we are not generating new data, just store NaN values
            elif only_load:
                loss_values[j, i] = np.NaN
                continue

            # If we are generating new data
            else:

                # Compile and train the model
                model.compile(optimizer='adam', 
                            loss=PNNLoss(),
                            metrics=['accuracy'])
                history = train_pnn(model, model_path=model_path, 
                                    epochs=epochs, steps_per_epoch=steps_per_epoch, 
                                    batch_size=batch_size, **parameters)

            # Identify the last value of loss
            losses = history.history['loss']
            loss_values[j, i] = losses[-1]

            # Generate one of the plots
            if extra_plots and (k == 10 and i == 6):
                pnn_analysis(model, **parameters)


    # Normalize loss values to be minimized at 0 by adding 51200
    loss_values -= min_batch_loss
    max_loss = np.max(loss_values)

    # Plot the loss vs phase restriction
    fig, ax = plt.subplots(1)
    for j, k in enumerate(k_vals):
        ax.plot(phase_restrictions, loss_values[j, :], label=f'{k} masks')

    ax.set_xticks([0, np.pi/2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3 \pi/2$', r'$2 \pi$'])
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xlabel('Phase Restriction')
    ax.set_ylabel('Measured Loss')
    ax.set_title(f'Training Loss for Vortex Mask')

    # Create the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)

    # Plot a line at 2 pi / k
    ax.plot([2*np.pi / k, 2*np.pi/k], [0, max_loss], ':k')

    # Display the plot
    plt.show()
