from nptyping import NDArray
import numpy as np
from numpy import min, max, meshgrid, pi, sqrt, exp, arange
from numpy.fft import fftfreq, fftshift, ifftshift


def compute_FSTF(input_field, output_field, dx, dy,
                 distance, wavelength,
                 k_space_filter: int = None):

    # Ensure the input and output fields have the same shape
    shape = input_field.shape
    if shape != output_field.shape:
        raise SyntaxError('Input and Output fields must have the same shape')

    # Calculate the x and y coordinates
    Ny, Nx = shape
    x = dx * (arange(Nx) - (Nx / 2))
    y = dy * (arange(Ny) - (Ny / 2))
    x, y = meshgrid(x, y)

    # Compute the free space transfer function for the given distance
    h = free_space_transfer_function(x, y, distance, wavelength)

    # Filter the transfer function
    if k_space_filter is not None:
        if k_space_filter < 1:
            radius = sqrt(x ** 2 + y ** 2)
            max_radius = max(radius)
            frequency_mask = radius < max_radius * k_space_filter
            h *= ifftshift(frequency_mask)

    # return the result
    return h


def free_space_transfer_function(x, y, z, wavelength, shift=False):
    """
    This is the free space transfer function used by Joel Carpenter

    :param x: NDArray
        X coordinate meshgrid

    :param y: NDArray
        Y coordinate meshgrid

    :param z: float
        Propagation distance in the z direction

    :param wavelength: float
        Wavelength of light

    :param shift: boolean
        Select whether or not to shift the k coordinates

    :return: NDArray
        Transfer function of free space
    """

    # Get x and y coordinates
    ny, nx = x.shape
    dx = x[0, 1] - x[0, 0]
    dy = y[1, 0] - y[0, 0]

    # Get k-space coordinates for the Fourier transform
    if shift:
        # kx = (arange(Nx) - (Nx / 2)) / (max(x) - min(x))
        kx = fftshift(fftfreq(nx, d=dx))
        ky = fftshift(fftfreq(ny, d=dy))
    else:
        kx = fftfreq(nx, d=dx)
        ky = fftfreq(ny, d=dy)
    kx, ky = meshgrid(kx, ky)

    # Compute the transfer function
    exponent = -1j * 2 * pi * sqrt(
        (wavelength ** (-2)) + (kx ** 2) + (ky ** 2))
    return exp(exponent * z)
