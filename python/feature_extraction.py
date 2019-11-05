import cv2 as cv
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_multilabel_classification

img_path = '/Users/crystalwang/Documents/test.png'
n_keypoints = 1000 #hyperparameter, need to tune

def get_feature_vector(img):
    # Get keypoints and feature descriptors
    sift = cv.xfeatures2d_SIFT.create(n_keypoints)
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def build_histogram(descriptor_list, cluster_alg, n_clusters):
    """Helper function/sub-routine that uses a fitted clustering algorithm
    and a descriptor list for an image to a histogram."""
    histogram = np.zeros(n_clusters)
    cluster_result = cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
        return histogram


def create_feature_matrix(img_path, descriptor_list, n_clusters=800):
    """Main function for creating a matrix of size N_images x n_clusters
    using SIFT and histogramming of the descriptors by a clustering
    algorithm."""
    # Make clustering algorithm
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(descriptor_list)
    # Get image files
    img_files = os.listdir(img_path)
    print(img_files)

    M = []
    for f in img_files:  # Iterate over all image files
        A = cv.imread(os.path.join(img_path, f))  # Read image
        kp, des = get_feature_vector(A)  # Get keypoints/descriptors from SIFT
        histogram = build_histogram(des, kmeans, n_clusters)
        M.append(histogram)  # Append to output matrix
    print(M.shape)

def main():
    # img = cv.imread(img_path)
    # img2 = img
    # # Get keypoints and descriptors from SIFT
    # kp, des = get_feature_vector(img)
    # print(len(kp))
    # print(des)
    # print(des.shape)
    # img2 = cv.drawKeypoints(img, kp, img2)
    # img2 = cv.drawKeypoints(img, kp, img2)
    # cv.imwrite("/Users/crystalwang/Documents/test_keypoints.png", img2)
    X , _ = make_multilabel_classification(random_state=5)
    print(X.shape)
    print(X)


if __name__ == "__main__":
    main()
