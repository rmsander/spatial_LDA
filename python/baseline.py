import numpy as np
import matplotlib as plt
from sklearn.cluster import KMeans
from skimage.transform import rescale, resize
from dataset import ADE20K, get_single_loader
from itertools import zip_longest
import os
from torchvision import transforms
import pickle

data_root = os.path.join(os.path.dirname(__file__), '../data')


NUM_KMEANS_CLUSTERS = 100

YAATEH_DATA_ROOT = "/Users/yaatehr/Programs/spatial_LDA/data/seg_data"
PICKLE_SAVE_RUN = False
IMAGE_MATRIX_PATH = os.path.join(data_root, "grayscale_img_matrix.pkl")


def pca(X):
    """
    Computes eigenvectors of the covariance matrix of X
    """
    m,n = X.shape[0], X.shape[1]
    xsquared = X.T @ X
    
    sigma = 1/m * xsquared
    print('entering SVD loop')

    
    U,S,V = np.linalg.svd(sigma)
    print('SVD loop complete')

    return U,S,V


def featureNormalize(X):
    """
    Returns a normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.
    Note that this operates on unraveled images in a matrix X (each image is a row)
    """
    mu = np.mean(X,axis=1)
    sigma = np.std(X,axis=1)
    print(mu)
    print(sigma)
    centeredX = (X.T- mu).T
    
    X_norm = centeredX/sigma[:, None]
    
    return X_norm, mu , sigma


def stack_images_rows_with_pad(list_of_images):
    """
    If/when we use a transform this won't be necessary
    """
    maxlen = max([len(x) for x in list_of_images])
    out = np.vstack([np.concatenate([b, np.zeros(maxlen-len(b))]) for b in list_of_images])
    print(out.shape)
    if PICKLE_SAVE_RUN:
        with open(IMAGE_MATRIX_PATH, 'wb') as f:
            pickle.dump(out, f)
    return out


# def dataloaderToMatrix(loader):
#     np.ndarray = 


pca_rescale_factor = .25


def resize_im_shape(img_shape, maxEdgeLen = 225):
    x,y = img_shape
    if x > y:
        maxEdge = x
    else:
        maxEdge = y

    scalefactor = maxEdgeLen/maxEdge
    remainderEdge = int(min(x,y) * scalefactor)
    if x > y:
        return maxEdgeLen, remainderEdge
    return remainderEdge, maxEdgeLen

def resize_im(im):
    return resize(im, resize_im_shape(im.shape), anti_aliasing=False)
    

def createFeatureVectors():
    grayscaleDataset = ADE20K(grayscale=True, root=YAATEH_DATA_ROOT, transform=resize_im)
    dataset = get_single_loader(grayscaleDataset, batch_size=1)
    print(grayscaleDataset.__getitem__(0)[0].shape)

    flattened_image_list = []
    label_list = []
    for step, (img, label) in enumerate(grayscaleDataset):
        if step > 100:
            break
        flattened_image_list.append(img.flatten())
        label_list.append(label)
    stacked_images = stack_images_rows_with_pad(flattened_image_list)
    normalized_images = featureNormalize(stacked_images)[0]
    print('entering PCA loop')
    U = pca(normalized_images)[0]
    print('entering KMEANS')

    kmeans = KMeans(n_clusters=NUM_KMEANS_CLUSTERS)
    print('fitting KMEANS')

    kmeans.fit(U)


    print('stacking vectors KMEANS')


    

    for i,d in enumerate(dataset):
        vstack = np.vstack([i for i in list(descriptor_list_dic.values()) if i is not None and i.shape[0] == n_keypoints])


        print(i)
        print(d)


createFeatureVectors()