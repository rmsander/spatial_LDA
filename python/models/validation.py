import numpy as np
from feature_extraction import get_feature_vector, build_histogram
from lda import LDA2
import pickle
from dataset import get_single_loader, ADE20K
import os
from skimage import io
import cv2 as cv


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
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)
    for idx, (image, label) in enumerate(grayscaleDatasetVal):
        letter = label[0]
        letter_path = os.path.join(dataset_path, letter)
        if not os.path.exists(letter_path):
            os.mkdir(letter_path)
        label_path = os.path.join(letter_path, label)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        img_path = grayscaleDataset.image_paths[idx]
        image_filename = img_path.split('/')[-1]
        complete_pathname = os.path.join(label_path, image_filename)
        print(complete_pathname)
        io.imsave(complete_pathname, image)

def main():
    dataset_path = "/home/yaatehr/programs/datasets/seg_data/images/dataset1_val/"
    make_directory_for_validation(dataset_path)
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
    prob_distrs_validation = {}
    letter_files = os.listdir(dataset_path)
    num_files = 0
    for letter in letter_files:
        label_path = os.listdir(dataset_path, letter)
        labels = os.listdir(label_path)
        for label in labels:
            singular_label_path = os.path.join(label_path, label)
            images = os.listdir(singular_label_path) 
            for f in images:
                if f[-3:] == 'jpg':
                    continue
                if num_files%100 == 0:
                    print(num_files)
                img = cv.imread(f)
                prediction = get_prediction_for_image(img, lda_model, kmeans_model)
                prob_distrs_validation[f] = prediction