import cv2 as cv
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_multilabel_classification
import pickle
from torchvision import transforms
from skimage import io
from train_cnn import get_model, resnet_transform

n_keypoints = 100 #hyperparameter, need to tune
n_cnn_keypoints = 4*49

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

def get_difference_histograms(hist1, hist2, metric="l2"):
    if metric == 'l2':
        return np.sum(np.square(hist2 - hist1))
    if metric == 'l1':
        return np.sum(np.abs(hist1-hist2))

def create_feature_matrix(img_path, n_clusters=80):
    """Main function for creating a matrix of size N_images x n_clusters
    using SIFT and histogramming of the descriptors by a clustering
    algorithm."""
    # Make clustering algorithm
    kmeans = KMeans(n_clusters=n_clusters)
    img_files = os.listdir(img_path)
    # print(img_files)
    print(len(img_files))
    with open("/home/yaatehr/programs/spatial_LDA/data/img_descriptors_dic1.pkl","rb") as f:
        descriptor_list_dic = pickle.load(f)
    # descriptor_list_dic = {}
    # for f in img_files:
    #     A = cv.imread(os.path.join(img_path, f)) # read image
    #     _, des = get_feature_vector(A)
    #     descriptor_list_dic[f]= des
    with open("/home/yaatehr/programs/spatial_LDA/data/img_descriptors_dic1.pkl", "wb") as f:
        pickle.dump(descriptor_list_dic, f)
    vstack = np.vstack([i for i in list(descriptor_list_dic.values()) if i is not None and i.shape[0] == n_keypoints])
    print(vstack.shape)
    kmeans.fit(vstack)

    # Get image files
    M = []
    num_files = 0
    for f in img_files:  # Iterate over all image files
        if num_files % 100 == 0:
            print(str(num_files)+" files processed")
        des = descriptor_list_dic[f]  # Get keypoints/descriptors from SIFT
        if des is None or des.shape[0] != n_keypoints:
            continue
        histogram = build_histogram(des, kmeans, n_clusters)
        
        M.append(histogram)  # Append to output matrix
        num_files += 1
    return M

def create_feature_matrix_cnn(img_path, model, n_clusters=80):
    kmeans = KMeans(n_clusters=n_clusters)
    img_files = os.listdir(img_path)
    print(len(img_files))
    with open("/home/yaatehr/programs/spatial_LDA/data/cnn_descriptors_dict01.pkl","rb") as f:
        descriptor_list_dic = pickle.load(f)
    #print(img_files)
    #descriptor_list_dic = {}
    #for f in img_files:
    #    test_img = io.imread(f)
    #    inputs = resnet_transform(test_img)
    #    inputs = inputs.unsqueeze(0)
    #    des = model(inputs).view(4*49,128).detach().numpy()
    #    descriptor_list_dic[f]= des
    #    del test_img
    box_path = "/home/yaatehr/programs/spatial_LDA/data/cnn_descriptors_dict01.pkl"
    local_path = "/Users/yaatehr/Programs/spatial_LDA/cnn_descriptors_dict01.pkl"
    with open(box_path, "wb") as f:
        pickle.dump(descriptor_list_dic, f)
    vstack = np.vstack([i for i in list(descriptor_list_dic.values()) if i is not None and i.shape[0] == n_cnn_keypoints])
    print(vstack.shape)
    kmeans.fit(vstack)

    # Get image files
    M = []
    num_files = 0
    for f in img_files:  # Iterate over all image files
        if num_files % 100 == 0:
            print(str(num_files)+" files processed")
        des = descriptor_list_dic[f]  # Get keypoints/descriptors from CNN
        if des is None or des.shape[0] != n_cnn_keypoints:
            continue
        histogram = build_histogram(des, kmeans, n_clusters)
        
        M.append(histogram)  # Append to output matrix
        num_files += 1
    return M

def main():
    dataset_path = "/home/yaatehr/programs/spatial_LDA/data/descriptors_test_1"
    # M = create_feature_matrix(dataset_path)
    model = get_model()
    CnnMatrix = create_feature_matrix_cnn(dataset_path, model)

    with open("/home/yaatehr/programs/spatial_LDA/data/cnn_feature_matrix", "wb") as f:
        pickle.dump(CnnMatrix, f)


if __name__ == "__main__":
    main()
