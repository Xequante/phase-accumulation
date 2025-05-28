import numpy as np
from src.propagation.transfer_functions import free_space_transfer_function


def free_space_propagate(wavefront, h=None,
                         wavelength=None, distance=None,
                         x=None, y=None, dx=None, dy=None):
    """
    Propagate a wavefront using the angular spectrum method (NumPy version).

    Parameters:
    wavefront : 2D ndarray
        Complex wavefront to propagate

    h : 2D ndarray or None
        Precomputed transfer function.
        If provided, x/y and other parameters are ignored.

    wavelength : float
        Wavelength of light (meters)

    distance : float
        Propagation distance (meters)

    x, y : 2D ndarrays
        Coordinate meshgrids (used to compute frequency axes)

    dx, dy : float
        Sampling periods in x and y directions
        (used if x and y are not provided)

    Returns:
    propagated_field : 2D ndarray
        Resulting wavefront after propagation
    """
    # Forward FFT of wavefront
    wavefront_ft = np.fft.fft2(wavefront)

    if h is None:
        if x is None or y is None:
            if dx is None or dy is None:
                raise SyntaxError('Must either input both x and y'
                                  ' or both dx and dy')

            Ny, Nx = wavefront.shape
            x = dx * (np.arange(Nx) - Nx / 2)
            y = dy * (np.arange(Ny) - Ny / 2)
            x, y = np.meshgrid(x, y)

        h = free_space_transfer_function(x, y, distance, wavelength)

    # Apply transfer function
    wavefront_ft *= h

    # Inverse FFT
    propagated = np.fft.ifft2(wavefront_ft)
    return propagated
