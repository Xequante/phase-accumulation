import tensorflow as tf


class HaltCallback(tf.keras.callbacks.Callback):
    def __init__(self, loss_threshold):
        super().__init__()
        self.loss_threshold = loss_threshold

    def on_epoch_end(self, epoch, logs=None):
        if self.loss_threshold is None:
            return
        if logs.get('loss') <= self.loss_threshold:
            print(f"\nReached loss value of {logs.get('loss')}, stopping training!")
            self.model.stop_training = True
