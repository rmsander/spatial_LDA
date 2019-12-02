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

