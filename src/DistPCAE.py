


"""
ABOUT: DISTANCE PRESERVING CONVOLUTIONAL AUTO-ENCODER
       for UNIVARIATE TIME SERIES.
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
                                     Flatten,
                                     Lambda,
                                     MaxPooling1D,
                                     Reshape,
                                     UpSampling1D)
from tensorflow_addons.losses.metric_learning import pairwise_distance as PD



# =============================================================================
# CLASS
# =============================================================================



class DistPCAE(AEBase):



    def __init__(self,
                 input_size,
                 n_layers,
                 latent_dim,
                 n_filters=16,
                 kernel_size=0.03,
                 activation='tanh',
                 optimizer='Adam',
                 lr=0.001,
                 seed=None,
                 loss_weights=[1., 1.],
                 **kwargs):

        self.input_size = int(input_size)
        self.n_layers = int(n_layers)
        self.latent_dim = int(latent_dim)

        self.n_filters = int(n_filters)
        self.kernel_size = kernel_size

        self.pooling = self.get_pooling()
        self.filters = self.get_filters()
        self.kernels = self.get_kernels()

        self.seed = seed

        self.activation = tf.keras.activations.get(activation)

        self.optimizer = getattr(tf.keras.optimizers, optimizer)(learning_rate=lr)

        self.encoder_input = Input((self.input_size,1), name='layer_0.in')
        self.decoder_input = Input((self.latent_dim,), name='layer_0.out')
        self.distance_matrix_input = Input((None,), name='distance_matrix')

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.encoded_input = self.encoder(self.encoder_input)
        self.dp_loss = Lambda(PD, name='dp_loss')(self.encoded_input)

        self.model = Model(inputs=[self.encoder_input, self.distance_matrix_input],
                           outputs=[self.decoder(self.encoded_input), self.dp_loss])

        self.model.compile(loss=['mse', 'mse'],
                           loss_weights=loss_weights,
                           optimizer=self.optimizer)

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

        model.add(Flatten(data_format='channels_last'))

        self.flat_dim = int(self.input_size / (sum(self.pooling)*2))

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

        model.add(Reshape((self.flat_dim,1)))

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



    def fit(self, X, distance_matrix, epochs, batch_size=None):
        if batch_size is None:
            batch_size = X.shape[0]

        def generator(X, dist_matrix, batch_size):
            nsamples = X.shape[0]
            while True:
                ix = tf.random.uniform(shape=(batch_size,),
                                       minval=0,
                                       maxval=nsamples,
                                       dtype=tf.int32)
                x = tf.gather(X, indices=ix, axis=0)
                dm = tf.gather_nd(dist_matrix, indices=tf.stack(tf.meshgrid(ix, ix), axis=-1))
                yield (x, dm), (x, dm)

        Data = tf.data.Dataset.from_generator(generator,
                                              ((tf.float32, tf.float32), (tf.float32, tf.float32)),
                                              ((tf.TensorShape([None,self.input_size,1]), tf.TensorShape([batch_size,batch_size])), (tf.TensorShape([None,self.input_size,1]), tf.TensorShape([batch_size,batch_size]))),
                                              args=[X, distance_matrix, batch_size])

        steps = int(tf.math.ceil(X.shape[0]/batch_size))
        self.model.fit(Data,
                       epochs=epochs,
                       shuffle=False,
                       steps_per_epoch=steps,
                       callbacks=self.call_backs,
                       validation_data=None,
                       verbose=False,
                       use_multiprocessing=False)



    def loss_history(self):
        history = self.loss_tracker.get_history()
        return {'total_loss': history.loss.values,
                'r_loss': history.Decoder_loss.values,
                'dp_loss': history.dp_loss_loss.values}



# =============================================================================
# MAIN
# =============================================================================



if __name__ == '__main__':



    import os
    import matplotlib.pyplot as plt
    from dissimilarity import dissimilarity
    from scipy.spatial.distance import cdist


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

    diss = 'DTW'
    D = dissimilarity()
    D = getattr(D, diss)
    _X = tf.squeeze(X_train_pos, axis=-1)
    DM = cdist(_X, _X, metric=D)
    DM = tf.linalg.normalize(DM, ord='euclidean')[0]


    # MODEL
    model = DistPCAE(input_size=X_train_pos.shape[1],
                     n_layers=5,
                     latent_dim=2,
                     optimizer='Adam',
                     activation='tanh',
                     lr=0.001)


    print('ENC structure: ', model.pooling)
    print('Training Samples: ', X_train_pos.shape[0])
    model.show_config(model.encoder)
    model.show_config(model.decoder)


    # FIT
    model.fit(X_train_pos,
              distance_matrix=DM,
              epochs=1000,
              batch_size=16)


    # PLOT LOSS
    loss = model.loss_history()
    plt.plot(loss['total_loss'], '-k', label='total')
    plt.plot(loss['r_loss'], '-r', label='rec')
    plt.plot(loss['dp_loss'], '-b', label='dist')
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
    plt.plot(X_reconstructed.ravel(), '-b')
    plt.title('Reconstruction (black=original - blue=reconstred)')
    plt.show()
    plt.close()



# =============================================================================
# THE END
# =============================================================================