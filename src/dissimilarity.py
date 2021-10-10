


__author__ = 'Stefano Mauceri'

__email__ = 'mauceri.stefano@gmail.com'



# =============================================================================
# IMPORT
# =============================================================================



import numba
import numpy as np
from scipy.stats import entropy
import scipy.spatial.distance as ssd
from statsmodels.tsa.stattools import acf
from scipy.stats import wasserstein_distance as wsd
from sklearn.metrics.pairwise import rbf_kernel, sigmoid_kernel



# =============================================================================
# FUNCTIONS working with NUMBA.JIT
# =============================================================================



@numba.njit
def dtw(a, b, window):

    length_a,  length_b = a.size, b.size
    C = np.full(shape=(length_a + 1, length_b + 1), fill_value=np.inf)
    C[0,0] = 0

    if window is None:

        for i in range(1, length_a+1):
            for j in range(1, length_b+1):
                C[i, j] = abs(a[i-1] - b[j-1])  + min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])
    else:

        for i in range(1, length_a+1):
            for j in range(1, length_b+1):
                if abs(i-j) <= window:
                    C[i, j] = abs(a[i-1] - b[j-1])  + min(C[i, j - 1], C[i - 1, j - 1], C[i - 1, j])

    return C[length_a, length_b]
#    gap = abs(length_a - length_b)
#    if gap <= window:
#        return C[length_a, length_b]
#    else:
#        if length_a > length_b:
#            length_a -= abs(gap-window)
#        else:
#            length_b -= abs(gap-window)
#        return C[length_a, length_b]



@numba.njit
def edr(a, b, eps):
    length_a, length_b = a.size, b.size
    C = np.zeros((length_a + 1, length_b + 1))

    for i in range(1, length_a+1):
        for j in range(1, length_b+1):
            C[i,j] = abs(a[i] - b[j])

    for i in range(1, length_a + 1):
        for j in range(1, length_b + 1):
            if C[i, j] < eps:
                C[i, j] = min(C[i, j - 1] + 1, C[i - 1, j] + 1, C[i - 1, j - 1] + 0)
            else:
                C[i, j] = min(C[i, j - 1] + 1, C[i - 1, j] + 1, C[i - 1, j - 1] + 1)
    return C[length_a, length_b]



@numba.njit
def hist1d(x, binx):
    return np.histogram(x, binx)[0]



@numba.njit
def msm(x, y, c_penalty):

    def C(a1, a2, b, c):
        if a2 <= a1 <= b or a2 >= a1 >= b:
            return c
        else:
            return c + min(abs(a1-a2), abs(a1-b))

    length_x, length_y = x.size, y.size
    D = np.zeros(shape=(length_x,length_y))

    D[0,0] = abs(x[0] - y[0])

    for i in range(1, length_x):
        D[i,0] = D[i-1, 0] + C(x[i], x[i-1], y[0], c_penalty)

    for i in range(1, length_y):
        D[0,i] = D[0, i-1] + C(y[i], x[0], y[i-1], c_penalty)

    for i in range(1, length_x):
        for j in range(1, length_y):
            D[i,j] = min(D[i-1,j-1] + abs(x[i] - y[j]),
                         D[i-1,j] + C(x[i], x[i-1], y[j], c_penalty),
                         D[i, j-1] + C(y[j], x[i], y[j-1], c_penalty))

    return D[length_x-1, length_y-1]



# =============================================================================
# CLASS
# =============================================================================


class dissimilarity(object):



    __all__ = [
              'autocorrelation',
              'chebyshev',
              'cityblock',
              'cosine',
              'DTW',
              'EDR',
              'euclidean',
              'kullback_leibler',
              'minkowski',
              'MSM',
              'gaussian',
              'sigmoid',
              'WD'
              ]



    def __init__(self):
        pass



    def autocorrelation(self, a, b, **kwargs):
        """
        In the original formulation [Galeano and Pena (2000)] authors use
        the squared Euclidean norm. Here we prefer to use the Euclidean
        norm becuase the squared Euclidean norm is not a metric.
        """
        lags = int(a.shape[0]) - 1
        coeff = np.geomspace(1, 0.01, lags)
        try:
            return self.euclidean(acf(a, nlags=lags)[1:] * coeff, acf(b, nlags=lags)[1:] * coeff)
        except:
            return 1E5



    def chebyshev(self, a, b, **kwargs):
        return ssd.chebyshev(a, b)



    def CID(self, a, b, **kwargs):
        """
        Complexity-Invariant Distance (CID)
        Batista, Wang & Keogh (2011).
        A Complexity-Invariant Distance Measure for Time Series.
        """
        ce_a = np.sqrt(np.sum(np.square(np.diff(a))) + 1e-9)
        ce_b = np.sqrt(np.sum(np.square(np.diff(b))) + 1e-9)
        return np.linalg.norm(a - b) * np.divide(np.maximum(ce_a, ce_b), np.minimum(ce_a, ce_b))



    def cityblock(self, a, b, **kwargs):
        return ssd.cityblock(a, b)



    def cosine(self, a, b, **kwargs):
        return ssd.cosine(a, b)



    def DTW(self, a, b, window=None, **kwargs):
        # Dynamic time warping
        return dtw(a, b, window, **kwargs)



    def EDR(self, a, b, eps, **kwargs):
        # Edit distance on real sequences
        return edr(a, b, eps, **kwargs)



    def euclidean(self, a, b, **kwargs):
        return ssd.euclidean(a, b)



    def kullback_leibler(self, a, b, **kwargs):
        ab = np.concatenate([a, b])
        bins = np.linspace(ab.min(), ab.max(), 10)
        a, b = hist1d(a, bins) + 1, hist1d(b, bins) + 1
        return entropy(a, b) + entropy(b, a)



    def minkowski(self, a, b, p, **kwargs):
        return ssd.minkowski(a, b, p)



    def MSM(self, a, b, c_penalty, **kwargs):
        # Move split merge
        return msm(a, b, c_penalty)



    def gaussian(self, a, b, **kwargs):
        # gamma = 1 / ts_length
        return rbf_kernel(a.reshape(1, -1), b.reshape(1, -1), gamma=None) * -1



    def sigmoid(self, a, b, **kwargs):
        # gamma = 1 / ts_length
        return sigmoid_kernel(a.reshape(1, -1), b.reshape(1, -1), gamma=None, coef0=0) * -1


    def WD(self, a, b, **kwargs):
        # Wasserstein distance
        return wsd(a, b)



# =============================================================================
# MAIN
# =============================================================================



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    np.random.seed(123)
    D = dissimilarity()
    tslength = 10
    a = np.random.randint(0, 90, size=(tslength))
    b = np.random.randint(0, 45, size=(tslength))

    print(D.DTW(a, b, 5))

    print(D.CID(a, b))


#    print(D.cityblock(a, b))
#    print('Window = 0: ', D.DTW(a,b, window=0))
#    print('Manhattan dist: ', ssd.cityblock(a, b))
#
#    dist = []
#    for w in range(0, 50):
#        dist.append(D.DTW(a,b, window=w))
#
#    plt.plot(dist, '-k')
#    plt.title('DTW distance per window size')
#    plt.xlabel('Window size')
#    plt.ylabel('Distance')
#    plt.show()
#    plt.close()



# =============================================================================
# END
# =============================================================================




