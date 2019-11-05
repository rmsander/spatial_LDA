import cv2 as cv
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_multilabel_classification

img_path = '/Users/crystalwang/Documents/test.png'
n_keypoints = 1000 #hyperparameter, need to tune

def get_feature_vector(img):
    sift = cv.xfeatures2d_SIFT.create(n_keypoints)
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


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
