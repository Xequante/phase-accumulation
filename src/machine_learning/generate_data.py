import numpy as np
from src.propagation.transfer_functions import free_space_transfer_function
from src.propagation.propagate import mplc_propagate, free_space_propagate


def generate_data(k, n, m, phase_restriction, 
                  batch_size: int = 32, 
                  wavelength: float = 500e-9,
                  distance: float = 0.05,
                  dx=10e-6, dy=10e-6):
    """
    Generate pairs of inputs and modulated outputs.
    """

    # Define the plane wave inputted into the system
    input_field = np.ones((n, m))

    # Generate the free space transfer function
    x = dx * (np.arange(m) - m/2)
    y = dy * (np.arange(n) - n/2)
    X, Y = np.meshgrid(x, y)
    h = free_space_transfer_function(X, Y, distance, wavelength)

    # Loop indefinitely, the generator never "ends"
    while True:  
        inputs = []
        outputs = []
        for i in range(batch_size):

            # Propagate input wavefront through a series of randomly generated masks
            wavefront = free_space_propagate(wavefront=input_field, h=h)
            masks = np.random.uniform(low=0, high=phase_restriction, size=(k,n,m))
            for i in range(k):
                wavefront = mplc_propagate(wavefront, masks[i], h=h)

            # Add results to inputs and outputs
            inputs.append(wavefront)
            outputs.append(masks)

        yield inputs, outputs
        