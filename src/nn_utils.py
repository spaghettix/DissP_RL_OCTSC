


"""
ABOUT: Some helper functions for neural networks.
"""


__author__ = 'Stefano Mauceri'
__email__ = 'mauceri.stefano@gmail.com'



# =============================================================================
# IMPORT
# =============================================================================



import numpy as np
import pandas as pd
from tensorflow import expand_dims
from tensorflow.keras.callbacks import Callback



# =============================================================================
# CLASS
# =============================================================================



class LossHistory(Callback):


    def on_train_begin(self, logs={}):
        self.container = []



    def on_epoch_end(self, batch, logs={}):
        self.container.append(logs)



    def get_history(self):
        return pd.DataFrame(self.container)



class AEBase(object):



    def __init__(self):
        pass



    def encode(self, X):
        return self.encoder(X)



    def decode(self, X):
        return self.decoder(X)



    def encode_decode(self, X):
        X = self.decoder(self.encoder(X))
        if len(X.shape) == 3:
            return np.squeeze(X, axis=-1)
        else:
            return X



    def load_existing_weigths(self, weights_path):
        self.model.load_weights(weights_path)



    def show_config(self, model):
        model.summary()
        for layer in model.layers:
            print(layer.get_config())
            print()



    def get_pooling(self):
        pooling = [0] * self.n_layers
        ix = self.n_layers // 2
        pooling[0] = 1
        pooling[ix] = 1
        return pooling



    def get_filters(self):
        return [self.n_filters] * self.n_layers



    def get_kernels(self):
        if type(self.kernel_size) is int:
            return [self.kernel_size] * self.n_layers
        else:
            input_size = self.input_size
            kernels = []
            for i in range(self.n_layers):
                k = np.floor(self.kernel_size * input_size)
                if k < 3:
                    k = 3
                kernels.append(int(k))
                if self.pooling[i] == 1:
                    input_size /= 2
            return kernels



    def check_input_conv(self, X):
        tsl = X.shape[1]
        r = tsl % 4
        if r == 0:
            return expand_dims(X, axis=2)
        else:
            return expand_dims(X[:, :-r], axis=2)



# =============================================================================
# THE END
# =============================================================================