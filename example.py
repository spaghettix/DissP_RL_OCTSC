


"""
ABOUT: EXAMPLE about REPRESENTATION LEARNING
and ONE-CLASS TIME SERIES CLASSIFICATION from the
paper Dissimilarity-Preserving Representation
Learning for One-Class Time Series Classification.
"""


__author__ = 'Stefano Mauceri'
__email__ = 'mauceri.stefano@gmail.com'



# =============================================================================
# IMPORT
# =============================================================================



import os
import numpy as np
import tensorflow as tf
from nn_utils import AEBase
import matplotlib.pyplot as plt
from src.DistPCAE import DistPCAE
from dissimilarity import dissimilarity
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors

tf.keras.backend.set_floatx('float32')



# =============================================================================
# EXAMPLE
# =============================================================================



# IMPORT DATA
p = os.path.abspath(os.path.join('data'))

dts = 'Plane'
class_ = 1
diss = 'DTW'

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

D = dissimilarity()
D = getattr(D, diss)
DM = cdist(X_train_pos, X_train_pos, metric=D)
DM = tf.linalg.normalize(DM, ord='euclidean')[0]

X_train_pos = tf.expand_dims(X_train_pos, axis=2)
X_train_neg = tf.expand_dims(X_train_neg, axis=2)
X_test_pos = tf.expand_dims(X_test_pos, axis=2)
X_test_neg = tf.expand_dims(X_test_neg, axis=2)


# MODEL
model = DistPCAE(input_size=X_train_pos.shape[1],
                 n_layers=3,
                 latent_dim=2,
                 optimizer='Adam',
                 activation='tanh',
                 lr=0.001)


# SHOW MODEL ARCHITECTURE / CONFIGURATION
# print('ENC structure: ', model.pooling)
# print('Training Samples: ', X_train_pos.shape[0])
# model.show_config(model.encoder)
# model.show_config(model.decoder)


# FIT
model.fit(X_train_pos,
          distance_matrix=DM,
          epochs=500,
          batch_size=4)


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
plt.plot(X_reconstructed.numpy().ravel(), '-b')
plt.title('Reconstruction (black=original - blue=reconstred)')
plt.show()
plt.close()


# ONE-CLASS TIME SERIES CLASSIFICATION
Classifier = NearestNeighbors(n_neighbors=1)
Classifier.fit(X_train_pos_enc)

# ENCODE TEST DATA AND PREPRARE TEST LABELS
X_test_enc = tf.expand_dims(X_test, axis=2)
X_test_enc = model.encode(X_test_enc)

Y_test = (Y_test == class_).astype(np.int8)

# GET TEST SCORES
Test_scores = Classifier.kneighbors(X_test_enc)[0] * -1
# Test scores are multiplied by -1 because the ROC curve expects that
# more is better while in terms of dissimilarities less is better.

# GET AUROC
fpr, tpr, _ = roc_curve(Y_test, Test_scores, pos_label=1)
AUROC = auc(fpr, tpr) * 100
print()
print('AUROC:', round(AUROC, 1))



# =============================================================================
# END
# =============================================================================