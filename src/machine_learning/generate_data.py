import numpy as np
import tensorflow as tf
from src.propagation.transfer_functions import free_space_transfer_function
from src.propagation.propagate import mplc_propagate, free_space_propagate


def generate_data(k, n, m, phase_restriction, 
                  input_field, h, batch_size: int = 32):
    """
    Generate pairs of inputs and modulated outputs.
    """

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

            # Split the wavefront into real and imaginary components
            split_wavefront = np.zeros((n, m, 2))
            split_wavefront[:, :, 0] = np.real(wavefront)
            split_wavefront[:, :, 1] = np.imag(wavefront)

            # Add results to inputs and outputs
            inputs.append(split_wavefront)
            outputs.append(np.copy(masks))

        # Convert to np arrays, then yeild the input and output data
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        yield inputs, tf.convert_to_tensor(outputs, dtype=tf.float32)
        