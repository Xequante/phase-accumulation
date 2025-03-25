from src.propagation.transfer_functions import free_space_transfer_function
from src.utilities.checkerboard import checkerboard

from numpy import ones, arange, meshgrid, pi, arctan2
from scipy.special import genlaguerre

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def parameters_dictionary(n: int = 60, 
                          m: int = 60, 
                          k: int = 3,
                          phase_restriction: float = 2*pi,
                          wavelength: float = 500e-9,
                          distance: float = 0.05,
                          dx: float = 10e-6,
                          dy: float = 10e-6,
                          input_index: int = 0,
                          output_index: int = 0) -> dict:
    """
    Creates a dictionary of key word arguments with any physical parameters for the system
    """

    # Generate the free space transfer function
    x = dx * (arange(m) - m/2)
    y = dy * (arange(n) - n/2)
    X, Y = meshgrid(x, y)
    h = free_space_transfer_function(X, Y, distance, wavelength)
    h = tf.convert_to_tensor(h, dtype=tf.complex64)

    # Define the input field
    if input_index == 0:
        input_field = ones((n, m))
    else:
        raise SyntaxError(f'"{input_index}" is not a valid entry for <input_index>')

    # Define the output field
    if output_index == 0:
        output_field = 2 * checkerboard(n, m, 12, 12) - 1
    elif output_index == 1:
        phase = arctan2(Y, X)
        output_field = np.exp(1j * phase)
    elif output_index == 2:
        output_field = lg_mode(X, Y, w0 = (n/6)*dx)
    else:
        raise SyntaxError(f'"{output_index}" is not a valid entry for <output_index>')
    
    # Calculate the minimum possible loss for this input/output
    min_loss = - np.sqrt(np.sum(np.abs(input_field)**2) * np.sum(np.abs(output_field)**2))

    # Show the target output field
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.abs(output_field))
    axs[1].imshow(np.angle(output_field))
    plt.show()

    # Create a dictionary with all the physical parameters of the system
    return {
        'n': n,
        'm': m,
        'k': k,
        'input_field': input_field,
        'output_field': output_field,
        'min_loss': min_loss,
        'wavelength': wavelength,
        'distance': distance,
        'dx': 10e-6,
        'dy': 10e-6,
        'h': h,
        'phase_restriction': phase_restriction
    }

def lg_mode(X, Y, 
            w0 = 10e-5, wavelength = 500e-9, 
            l:int = 2, p:int = 1, 
            z: float = 0.):
    """
    l - Azimuthal Index
    p - Radial Index
    z - Distance from beam waist
    """
    
    # Grid for calculations
    r = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    # Parameters
    k = 2 * np.pi / wavelength

    # Define beam waist at z
    zR = np.pi * w0**2 / wavelength
    wz = w0 * np.sqrt(1 + (z / zR) ** 2)

    # Laguerre polynomial
    L_p_l = genlaguerre(p, abs(l))

    # Laguerre-Gaussian beam intensity profile
    Rz = z if z != 0 else np.inf  # Avoid division by zero
    zeta = np.arctan(z / zR)
    norm_factor = (np.sqrt(2) * r / wz) ** abs(l)
    amplitude = norm_factor * L_p_l(2 * r**2 / wz**2) * np.exp(-r**2 / wz**2)
    phase = np.exp(1j * l * phi) * np.exp(-1j * (2 * p + abs(l) + 1) * zeta)

    # Return amplitude * phase
    return amplitude * phase