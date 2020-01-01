import numpy as np
import os

# Hyperparameters
NUM_KMEANS_CLUSTERS = 100

# Get relevant file paths and directories
data_root = os.path.join(os.path.dirname(__file__), '../data')
YAATEH_DATA_ROOT = "/Users/yaatehr/Programs/spatial_LDA/data/seg_data"
PICKLE_SAVE_RUN = False
IMAGE_MATRIX_PATH = os.path.join(data_root, "grayscale_img_matrix.pkl")


def pca(X):
    """
    Computes eigenvectors of the covariance matrix of X
    """
    print('entering PCA function call')

    m, n = X.shape[0], X.shape[1]
    xsquared = X.T @ X

    sigma = 1 / m * xsquared
    print('calling svd')

    U, S, V = np.linalg.svd(sigma)
    print('PCA function call complete')

    return U, S, V


def featureNormalize(X):
    """
    Returns a normalized version of X where the mean value of each feature is
    0 and the standard deviation is 1.
    Note that this operates on unraveled images in a matrix X (each image is
    a row)
    """
    mu = np.mean(X, axis=1)
    sigma = np.std(X, axis=1)
    centeredX = (X.T - mu).T

    X_norm = centeredX / sigma[:, None]

    return X_norm
