import cv2 as cv
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier

img_path = '/Users/crystalwang/Documents/test.png'

def get_feature_vector(img):
    sift = cv.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def main():
    img = cv.imread(img_path)
    img2 = img
    kp, des = get_feature_vector(img)
    print(kp)
    print(des)
    img2=cv.drawKeypoints(img,kp,img2)
    cv.imwrite("/Users/crystalwang/Documents/test_keypoints.png", img2)
    

if __name__ == "__main__":
    main()