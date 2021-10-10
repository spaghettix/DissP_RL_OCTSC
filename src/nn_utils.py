


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
        return self.decoder(self.encoder(X))



    def show_config(self, model):
        model.summary()
        for layer in model.layers:
            print(layer.get_config())
            print()



    def get_pooling(self):
        pooling = np.zeros((self.n_layers,))
        ix = self.n_layers // 2
        pooling[0] = 1
        pooling[ix] = 1
        return pooling



    def get_filters(self):
        _filters = [32, 64, 96]
        base = np.array_split([1] * self.n_layers, 3)
        return np.concatenate([i*f for i,f in zip(base, _filters)])



    def check_input_conv(self, X):
        tsl = X.shape[1]
        if tsl % 4 == 0:
            return X
        else:
            for i in [1, -1, 2, -2]:
                if (tsl + i) % 4 == 0:
                    return X[:, :-abs(i)]



# =============================================================================
# END
# =============================================================================