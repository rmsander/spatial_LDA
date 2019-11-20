import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans
from dataset import ImageDataset, get_single_loader
from itertools import zip_longestgi

NUM_KMEANS_CLUSTERS = 100

YAATEH_DATA_ROOT = "/Users/yaatehr/Programs/spatial_LDA/data/seg_data"


def pca(X):
    """
    Computes eigenvectors of the covariance matrix of X
    """
    m,n = X.shape[0], X.shape[1]
    
    sigma = 1/m * X.T @ X
    
    U,S,V = np.linalg.svd(sigma)
    
    return U,S,V


def featureNormalize(X):
    """
    Returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.
    """
    mu = np.mean(X,axis=0)
    sigma = np.std(X,axis=0)
    
    X_norm = (X - mu)/sigma
    
    return X_norm, mu , sigma


def stack_images_rows_with_pad(list_of_images):
    func = lambda x: np.array(list(zip_longest(*x, fillvalue=0))) # applied row wise
    return np.array(list(map(func, zip(*list_of_images)))).transpose(2,0,1)


def dataloaderToMatrix(loader):
    np.ndarray = 


def createFeatureVectors():
    grayscaleDataset = ImageDataset(grayscale=True, root=YAATEH_DATA_ROOT)
    dataset = get_single_loader(grayscaleDataset, batch_size=1)
    

    for i,d in enumerate(dataset):
        vstack = np.vstack([i for i in list(descriptor_list_dic.values()) if i is not None and i.shape[0] == n_keypoints])


        print(i)
        print(d)


createFeatureVectors()