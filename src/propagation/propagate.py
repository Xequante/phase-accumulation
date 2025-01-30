from numpy.fft import fft2, ifft2
from numpy import exp, arange, meshgrid
from src.propagation.transfer_functions import free_space_transfer_function
import tensorflow as tf


def mplc_propagate(wavefront, mask, **kwargs):
    """
    Takes an wavefront and phase mask and finds the resulting field after
    propagating some distance

    :param wavefront:
        The wavefront incident on the phase mask

    :param mask:
        The phase mask (values between 0 and 2pi)

    :param **kwargs
        parameters for the free space propagation
    """

    # Compute the modulated wavefront
    modulated_wavefront = tf.cast(wavefront, tf.complex64) * tf.exp(1j * tf.cast(mask, tf.complex64))

    # Compute the free space propagation
    return free_space_propagate(modulated_wavefront, **kwargs)


def free_space_propagate(wavefront, h=None,
                         wavelength=None, distance=None,
                         x=None, y=None, dx=None, dy=None):
    """
    Takes an wavefront and phase mask and finds the resulting field after
    propagating some distance

    :param wavefront:
        The wavefront incident on the phase mask

    :param h: NDArray
        Free space transfer function.
        If this is inputted, none of the other kwargs are necessary

    :param wavelength:
        Wavelength of the light

    :param distance:
        Distance of propagation

    :param x: NDArray
        x coordinate meshgrid

    :param y: NDArray
        y coordinate meshgrid

    :param dx: float
        Sampling period in the x direction

    :param dy: float
        Sampling period in the y direction

    :return:
        Resulting wavefront
    """

    # Take the Fourier transform of the wavefront
    wavefront_ft = tf.signal.fft2d(wavefront)

    # Check for pre-inputted h
    if h is None:
        # Compute x and y if necessary
        if x is None or y is None:
            if dx is None or dy is None:
                raise SyntaxError('Must either input both x and y '
                                  'or both dx and dy')
            Ny, Nx = wavefront.shape
            x = dx * (arange(Nx) - (Nx / 2))
            y = dy * (arange(Ny) - (Ny / 2))
            x, y = meshgrid(x, y)

        # Obtain the free space transfer function
        h = free_space_transfer_function(x, y, distance, wavelength)

    # Multiply by the free space transfer function
    wavefront_ft *= h

    # Inverse Fourier Transform
    return tf.signal.ifft2d(wavefront_ft)
