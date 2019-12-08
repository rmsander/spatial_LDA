import numpy as np
from feature_extraction import get_feature_vector, build_histogram
from lda import LDA2
import pickle
from dataset import get_single_loader, ADE20K

def get_prediction_for_image(img, lda_model, kmeans_model):
    kp, des = get_feature_vector(img)
    feature = build_histogram(des, kmeans_model,kmeans_model.cluster_centers_.shape[0])
    predictions = lda_model.transofmr(np.reshape(feature, (1, feature.size)))
    return predictions

def make_directory_for_validation(dataset_path):
    training_path = '/home/yaatehr/programs/datasets/seg_data/images/training'
    validation_path = '/home/yaatehr/programs/datasets/seg_data/images/validation'
    grayscaleDataset = ADE20K(grayscale=True, root=training_path, transform = None, useStringLabels=True, randomSeed=49)
    mostCommonLabels = list(map(lambda x: x[0], grayscaleDataset.counter.most_common(25)))
    grayscaleDatasetVal = ADE20K(grayscale=True, root=validation_path, transform = None, useStringLabels=True, randomSeed=49)
    grayscaleDatasetVal.selectSubset(mostCommonLabels,normalizeWeights=True)
    

def main():
    n_clusters = 100
    n_topics = 20
    n_keypoints = 10
    sift_feature_path = "/home/yaatehr/programs/spatial_LDA/data/sift_feature_matrix_%s_keypoints_%s_clusters" %(n_keypoints, n_clusters)
    kmeans_path = "/home/yaatehr/programs/spatial_LDA/data/kmeans_%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints)
    validation_img_path = ""
    with open(sift_feature_path, 'rb') as f:
        M = pickle.load(f)
    with open(kmeans_path, 'rb') as f:
        kmeans_model = pickle.load(f)
    lda = LDA2("", sift_feature_path, n_topics = n_topics)
    lda_model = lda.off_the_shelf_LDA()

    prediction = get_prediction_for_image(img, lda_model, kmeans_model)
