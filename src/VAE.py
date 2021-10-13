


"""
ABOUT: DENSE VARIATIONAL AUTO-ENCODER
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
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau



# =============================================================================
# CLASS
# =============================================================================



class VAE(AEBase):



    def __init__(self,
                 input_size,
                 n_layers,
                 latent_dim=None,
                 structure=None,
                 activation='tanh',
                 optimizer='Adam',
                 lr=0.001,
                 seed=None,
                 loss_weights=[1., 1.],
                 **kwargs):

        input_size = int(input_size)

        if latent_dim is None:
            latent_dim = int(tf.math.ceil(tf.math.sqrt(tf.convert_to_tensor(input_size, dtype='float32'))))

        if structure is None:
            structure = np.geomspace(input_size, latent_dim, n_layers+1).astype(int)
            if structure[0] != input_size:
                structure[0] = input_size
            if structure[-1] != latent_dim:
                structure[-1] = latent_dim

        self.structure = structure

        self.seed = seed

        self.activation = tf.keras.activations.get(activation)

        self.optimizer = getattr(tf.keras.optimizers, optimizer)(learning_rate=lr)

        self.encoder_input = Input((input_size,), name='layer_0.in')
        self.decoder_input = Input((self.structure[-1],), name='layer_0.out')

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.model = Model(inputs=self.encoder_input,
                           outputs=[self.decoder(self.encoder(self.encoder_input)),
                                    self.encoder(self.encoder_input)])

        self.model.compile(loss=['mse', self.KLD],
                           loss_weights=[1., 1.],
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

        model = Sequential(name=f'Encoder')

        for i, d in enumerate(self.structure[1:], start=1):

            if i == len(self.structure)-1:
                act = None
                d *= 2
            else:
                act = self.activation

            model.add(Dense(d,
                      activation=act,
                      use_bias=True,
                      kernel_initializer={'class_name':'glorot_uniform',
                                          'config':{'seed':self.seed}},
                      bias_initializer='zeros',
                      name=f'layer_{i}.in'))

        model.add(Lambda(self.reparameterize, name='mean_logvar_z.in'))

        return model



    def reparameterize(self, X):
        mean, logvar = tf.split(X, num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=tf.shape(mean))
        return tf.stack([mean, logvar, eps * tf.exp(logvar * .5) + mean])



    def build_decoder(self):

        model = Sequential(name='Decoder')

        model.add(Lambda(lambda x: tf.unstack(x)[-1], name='mean_logvar_z.out'))

        structure = list(reversed(self.structure))
        for i, d in enumerate(structure[1:], start=1):

            if i == len(self.structure)-1:
                act = None
            else:
                act = self.activation

            model.add(Dense(d,
                      activation=act,
                      use_bias=True,
                      kernel_initializer={'class_name':'glorot_uniform',
                                          'config':{'seed':self.seed}},
                      bias_initializer='zeros',
                      name=f'layer_{i}.out'))

        return model



    def fit(self, X, epochs, batch_size=None):
        if batch_size is None:
            batch_size = X.shape[0]

        self.model.fit(x=X,
                       y=(X, X),
                       epochs=epochs,
                       shuffle=True,
                       callbacks=self.call_backs,
                       validation_data=None,
                       verbose=False)



    def KLD(self, y_true, y_pred):
        mean, logvar, z = tf.unstack(y_pred)
        return - 0.5 * tf.reduce_mean(logvar - tf.square(mean) - tf.exp(logvar) + 1)



    def encode(self, X):
        return tf.unstack(self.encoder(X))[-2]



    def loss_history(self):
        history = self.loss_tracker.get_history()
        return {'total_loss': history.loss.values,
                'r_loss': history.Decoder_loss.values,
                'kld_loss': history.Encoder_loss.values}



# =============================================================================
# MAIN
# =============================================================================



if __name__ == '__main__':



    import os
    import matplotlib.pyplot as plt

    # IMPORT DATA
    p = os.path.abspath(os.path.join('..', 'DATA'))

    dts = 'Plane'
    class_ = 1

    X_train = np.load(f'{p}/{dts}/{dts}_X_TRAIN.npy').astype(np.float32)
    Y_train = np.load(f'{p}/{dts}/{dts}_Y_TRAIN.npy')
    X_train = tf.linalg.normalize(X_train, axis=1, ord='euclidean')[0]

    X_train_pos = X_train[(Y_train == class_)]
    X_train_neg = X_train[(Y_train != class_)]

    X_test = np.load(f'{p}/{dts}/{dts}_X_TEST.npy').astype(np.float32)
    Y_test = np.load(f'{p}/{dts}/{dts}_Y_TEST.npy')
    X_test = tf.linalg.normalize(X_test, axis=1, ord='euclidean')[0]

    X_test_pos = X_test[(Y_test == class_)]
    X_test_neg = X_test[(Y_test != class_)]


    # MODEL
    model = VAE(input_size=X_train_pos.shape[1],
                n_layers=2,
                latent_dim=2,
                optimizer='Adam',
                activation='tanh',
                lr=0.001)


    print('ENC structure: ', model.structure)
    print('Training Samples: ', X_train_pos.shape[0])

    model.show_config(model.encoder)
    model.show_config(model.decoder)


    # FIT
    model.fit(X_train_pos,
              epochs=500,
              batch_size=16)


    # PLOT LOSS
    loss = model.loss_history()
    plt.plot(loss['total_loss'], '-k', label='total')
    plt.plot(loss['r_loss'], '-r', label='rec')
    plt.plot(loss['kld_loss'], '-b', label='kld')
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
    X_original = X_train_pos[1].numpy().reshape(1,-1)
    X_reconstructed = model.encode_decode(X_original)
    plt.plot(X_original.ravel(), '-k')
    plt.plot(X_reconstructed.numpy().ravel(), '-b')
    plt.title('Reconstruction (black=original - blue=reconstred)')
    plt.show()
    plt.close()



# =============================================================================
# END
# =============================================================================


