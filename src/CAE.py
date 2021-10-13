


"""
ABOUT: CONVOLUTIONAL AUTO-ENCODER
"""


__author__ = 'Stefano Mauceri'
__email__ = 'mauceri.stefano@gmail.com'



# =============================================================================
# IMPORT
# =============================================================================



import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
try:
    from .nn_utils import AEBase, LossHistory
except:
    from nn_utils import AEBase, LossHistory
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.layers import (Conv1D,
                                     Dense,
                                     Input,
                                     Lambda,
                                     MaxPooling1D,
                                     UpSampling1D)



# =============================================================================
# CLASS
# =============================================================================



class CAE(AEBase):



    def __init__(self,
                 input_size,
                 n_layers,
                 kernel_size=5,
                 latent_dim=None,
                 activation='tanh',
                 optimizer='Adam',
                 lr=0.001,
                 seed=None,
                 **kwargs):

        self.input_size = int(input_size)
        self.n_layers = n_layers

        if latent_dim is None:
            latent_dim = tf.math.ceil(tf.math.sqrt(tf.convert_to_tensor(input_size, dtype='float32')))
        self.latent_dim = int(latent_dim)

        self.pooling = self.get_pooling()
        self.filters = self.get_filters()
        self.kernels = np.zeros_like(self.filters) + kernel_size
        self.seed = seed

        self.activation = tf.keras.activations.get(activation)

        self.optimizer = getattr(tf.keras.optimizers, optimizer)(learning_rate=lr)

        self.encoder_input = Input((self.input_size, 1), name='layer_0.in')
        self.decoder_input = Input((self.latent_dim,), name='layer_0.out')

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.model = Model(inputs=self.encoder_input,
                           outputs=self.decoder(self.encoder(self.encoder_input)))

        self.model.compile(loss='mse', optimizer=self.optimizer)

        self.loss_tracker = LossHistory()

        self.lr_tracker = ReduceLROnPlateau(monitor='loss',
                                            factor=.5,
                                            patience=100,
                                            min_delta=0.0001,
                                            min_lr=0.0001,
                                            verbose=False)

        self.call_backs = [self.loss_tracker, self.lr_tracker]



    def build_encoder(self):

        model = Sequential(name='Encoder')
        ix = 0
        for i, p in enumerate(self.pooling):
            ix += 1
            model.add(Conv1D(filters=self.filters[i],
                             kernel_size=(self.kernels[i],),
                             strides=1,
                             padding='same',
                             data_format='channels_last',
                             activation=self.activation,
                             use_bias=True,
                             kernel_initializer={'class_name':'glorot_uniform',
                                                 'config':{'seed':self.seed}},
                             bias_initializer='zeros',
                             name=f'layer_{ix}.conv.in'))

            if p:
                ix += 1
                model.add(MaxPooling1D(pool_size=2,
                                       strides=None,
                                       padding='same',
                                       data_format='channels_last',
                                       name=f'layer_{ix}.pool.in'))

        model.add(Conv1D(filters=1,
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         data_format='channels_last',
                         activation=self.activation,
                         use_bias=True,
                         kernel_initializer={'class_name':'glorot_uniform',
                                             'config':{'seed':self.seed}},
                         bias_initializer='zeros',
                         name=f'layer_{ix+1}.1x1conv.in'))

        self.flat_dim = int(np.floor(self.input_size / (sum(self.pooling)*2)))

        model.add(Lambda(lambda x:tf.squeeze(x, axis=-1)))

        model.add(Dense(self.latent_dim,
                        activation=None,
                        use_bias=True,
                        kernel_initializer={'class_name':'glorot_uniform',
                                            'config':{'seed':self.seed}},
                        bias_initializer='zeros',
                        name=f'layer_{ix+3}.dense.in'))

        return model



    def build_decoder(self):

        model = Sequential(name='Decoder')

        model.add(Dense(self.flat_dim,
                        activation=self.activation,
                        use_bias=True,
                        kernel_initializer={'class_name':'glorot_uniform',
                                            'config':{'seed':self.seed}},
                        bias_initializer='zeros',
                        name='layer_1.dense.out'))

        model.add(Lambda(lambda x:tf.expand_dims(x, axis=-1)))

        pooling = np.flip(self.pooling)
        filters = np.flip(self.filters)
        kernels = np.flip(self.kernels)
        ix = 2
        for i, p in enumerate(pooling):
            ix += 1

            if p:
                model.add(UpSampling1D(size=2, name=f'layer_{ix}.unpool.out'))
                ix += 1

            model.add(Conv1D(filters=filters[i],
                             kernel_size=(kernels[i],),
                             strides=1,
                             padding='same',
                             data_format='channels_last',
                             activation=self.activation,
                             use_bias=True,
                             kernel_initializer={'class_name':'glorot_uniform',
                                                 'config':{'seed':self.seed}},
                             bias_initializer='zeros',
                             name=f'layer_{ix}.conv.out'))

        model.add(Conv1D(filters=1,
                         kernel_size=1,
                         strides=1,
                         padding='same',
                         data_format='channels_last',
                         activation=None,
                         use_bias=True,
                         kernel_initializer={'class_name':'glorot_uniform',
                                             'config':{'seed':self.seed}},
                         bias_initializer='zeros',
                         name=f'layer_{ix+1}.1x1conv.out'))

        return model



    def fit(self, X, epochs, batch_size=None):
        if batch_size is None:
            batch_size = X.shape[0]

        self.model.fit(x=X,
                       y=X,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       callbacks=self.call_backs,
                       validation_data=None,
                       verbose=False)



    def loss_history(self):
        history = self.loss_tracker.get_history()
        return history.loss.values



# =============================================================================
# MAIN
# =============================================================================



if __name__ == '__main__':



    import os
    import matplotlib.pyplot as plt


    # IMPORT DATA
    p = os.path.abspath(os.path.join('..', 'data'))

    dts = 'Plane'
    class_ = 1

    X_train = np.load(f'{p}/{dts}/{dts}_X_TRAIN.npy').astype(np.float32)
    X_train = AEBase().check_input_conv(X_train)
    Y_train = np.load(f'{p}/{dts}/{dts}_Y_TRAIN.npy')
    X_train = tf.linalg.normalize(X_train, axis=1, ord='euclidean')[0]

    X_train_pos = X_train[(Y_train == class_)]
    X_train_neg = X_train[(Y_train != class_)]

    X_test = np.load(f'{p}/{dts}/{dts}_X_TEST.npy').astype(np.float32)
    X_test = AEBase().check_input_conv(X_test)
    Y_test = np.load(f'{p}/{dts}/{dts}_Y_TEST.npy')
    X_test = tf.linalg.normalize(X_test, axis=1, ord='euclidean')[0]

    X_test_pos = X_test[(Y_test == class_)]
    X_test_neg = X_test[(Y_test != class_)]

    X_train_pos = tf.expand_dims(X_train_pos, axis=2)
    X_train_neg = tf.expand_dims(X_train_neg, axis=2)
    X_test_pos = tf.expand_dims(X_test_pos, axis=2)
    X_test_neg = tf.expand_dims(X_test_neg, axis=2)


    # MODEL
    model = CAE(input_size=X_train_pos.shape[1],
                n_layers=4,
                latent_dim=2,
                optimizer='Adam',
                activation='tanh',
                lr=0.001)


    print('Training Samples: ', X_train_pos.shape[0])

    model.show_config(model.encoder)
    model.show_config(model.decoder)


    # FIT
    model.fit(X_train_pos,
              epochs=500,
              batch_size=4)


    # PLOT LOSS
    loss = model.loss_history()
    plt.plot(loss, '-k', label='total')
    plt.title('Training Loss')
    plt.legend()
    plt.show()
    plt.close()


    # PLOT LATENT REPRESENTATION - TRAINING DATA
    X_train_pos_enc = model.encode(X_train_pos)
    X_train_neg_enc = model.encode(X_train_neg)
    plt.scatter(X_train_neg_enc[:, 0],
                X_train_neg_enc[:, 1],
                c='k', marker='o', alpha=0.3)
    plt.scatter(X_train_pos_enc[:, 0],
                X_train_pos_enc[:, 1],
                c='r', marker='o', alpha=0.3)
    plt.title('Training Data - Latent Space (red=positive - black=negative)')
    plt.show()
    plt.close()


    # PLOT LATENT REPRESENTATION - TEST DATA
    X_test_pos_enc = model.encode(X_test_pos)
    X_test_neg_enc = model.encode(X_test_neg)
    plt.scatter(X_test_neg_enc[:, 0],
                X_test_neg_enc[:, 1],
                c='k', marker='o', alpha=0.3)
    plt.scatter(X_test_pos_enc[:, 0],
                X_test_pos_enc[:, 1],
                c='b', marker='o', alpha=0.3)
    plt.title('Test Data - Latent Space (blue=positive - black=negative)')
    plt.show()
    plt.close()


    # PLOT RECONSTRUCTION
    X_original = X_train_pos[1].numpy().reshape(1,-1,1)
    X_reconstructed = model.encode_decode(X_original)
    plt.plot(X_original.ravel(), '-k')
    plt.plot(X_reconstructed.numpy().ravel(), '-b')
    plt.title('Reconstruction (black=original - blue=reconstred)')
    plt.show()
    plt.close()



# =============================================================================
# END
# =============================================================================