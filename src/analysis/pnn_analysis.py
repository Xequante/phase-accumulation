import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.propagation.propagate import mplc_propagate
from mpl_toolkits.axes_grid1 import make_axes_locatable


def pnn_analysis(model, input_field=None, output_field=None, 
                 n=40, m=40, k=None, phase_restriction=2*np.pi, **kwargs):

    # Extract Weights
    weights = model.get_weights()

    # Evaluate the prediction
    model_input = tf.reshape(input_field, (-1, n, m))
    prediction = model.predict(model_input)
    prediction = tf.squeeze(prediction)

    # Perform the propagation
    wavefront = input_field
    for i in range(k):
        phases = phase_restriction * tf.math.sin(weights[i]) / 2
        wavefront = mplc_propagate(wavefront, phases, **kwargs)

    # Plot a side by side
    fig, axs = plt.subplots(3, 2)

    # Target amplitude and phase
    ax = axs[0, 0]
    im = ax.imshow(tf.abs(output_field))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax = axs[0, 1]
    im = ax.imshow(tf.math.angle(output_field))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Prediction amplitude and phase
    ax = axs[1, 0]
    im = ax.imshow(tf.abs(prediction))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax = axs[1, 1]
    im = ax.imshow(tf.math.angle(prediction))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Ouput amplitude and phase
    ax = axs[2, 0]
    im = ax.imshow(tf.abs(wavefront))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax = axs[2, 1]
    im = ax.imshow(tf.math.angle(wavefront))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Plot the resulting phase masks
    num_rows = int(np.floor(np.sqrt(k)))
    num_cols = int(np.floor(np.sqrt(k)))
    if num_rows * num_cols < k:
        num_cols += 1
    if num_rows * num_cols < k:
        num_rows += 1
    fig2, axs2_grid = plt.subplots(num_rows, num_cols)
    axs2 = np.ravel(axs2_grid)
    for i in range(k):
        ax = axs2[i]
        phases = phase_restriction * tf.math.sin(weights[i]) / 2
        im = ax.imshow(phases)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

    # Display the plots
    plt.show()
    