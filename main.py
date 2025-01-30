from src.machine_learning.nn import create_neural_net
from src.machine_learning.compile import mplc_compile
from src.machine_learning.training import train_model
from src.propagation.transfer_functions import free_space_transfer_function
from numpy import ones, arange, meshgrid, pi


def main():

    # Generate a dictionary with necessary physical parameters
    parameters = parameters_dictionary(phase_restriction=2*pi)

    # Define a path where the model will be saved
    model_path = 'resources/neural_nets/start_model.h5'

    # Create and train the model
    train_new_model(model_path, **parameters)
    # train_loaded_model(model_path, **parameters)


def train_new_model(model_path, **kwargs):
    model = create_neural_net(**kwargs) 
    mplc_compile(model, **kwargs)
    train_model(model, model_path=model_path, **kwargs)


def train_loaded_model(model_path, **kwargs):
    model = create_neural_net(**kwargs)
    mplc_compile(model, **kwargs)
    model.load_weights(model_path)
    train_model(model, model_path=model_path, **kwargs)


def parameters_dictionary(n: int = 20, 
                          m: int = 20, 
                          phase_restriction: float = 2*pi,
                          wavelength: float = 500e-9,
                          distance: float = 0.05,
                          dx: float = 10e-6,
                          dy: float = 10e-6) -> dict:
    """
    Creates a dictionary of key word arguments with any physical parameters for the system
    """

    # Generate the free space transfer function
    x = dx * (arange(m) - m/2)
    y = dy * (arange(n) - n/2)
    X, Y = meshgrid(x, y)
    h = free_space_transfer_function(X, Y, distance, wavelength)

    # Create a dictionary with all the physical parameters of the system
    return {
        'n': n,
        'm': m,
        'input_field': ones((n, m)),
        'wavelength': wavelength,
        'distance': distance,
        'dx': 10e-6,
        'dy': 10e-6,
        'h': h,
        'phase_restriction': phase_restriction
    }


if __name__ == '__main__':
    main()