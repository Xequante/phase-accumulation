from src.machine_learning.nn import create_neural_net
from src.machine_learning.training import train_model


def main():
    model = create_neural_net()
    model_path = '/resources/neural_nets/start_model'
    train_model(model, model_path=model_path)