import cv2 as cv
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_multilabel_classification
import scipy
from scipy.special import kl_div as KL
import pickle
from torchvision import transforms
from skimage import io
from train_cnn import get_model, resnet_transform
import matplotlib.pyplot as plt
import argparse
from dataset import get_single_loader, ADE20K

n_keypoints = 50  # hyperparameter, need to tune
n_cnn_keypoints = 4 * 49
n_clusters = 50  # also need to tune this


def get_feature_vector(img):
    # Get keypoints and feature descriptors
    sift = cv.xfeatures2d_SIFT.create(n_keypoints)
    kp, des = sift.detectAndCompute(img, None)
    print(kp)
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
    """Helper function/sub-routine to compute the distance between two
    distributions."""
    if metric == 'l2':  # Euclidean distance
        return np.sum(np.square(hist2 - hist1))
    if metric == 'l1':  # Norm distance
        return np.sum(np.abs(hist1 - hist2))
    if metric == 'kl':  # Symmetric KL Divergence
        return 0.5 * np.sum(KL(hist1, hist2)) + 0.5 * np.sum(KL(hist2, hist1))


def evaluate_kmeans(descriptor_list, kmeans, n_clusters, metric="l2"):
    """Evaluation function for computing the k-means cluster distributions
    for images in the same ground truth classes.  Uses the descriptor list
    and clustering algorithm produced by the "create_feature_matrix" below.

    The distances can be indexed by their labels; i.e. each label key maps to a
    float distance value.
    """

    # Get files and directory
    label_dir = "/home/yaatehr/programs/datasets/seg_data/images/training/"
    label_letters = os.listdir(label_dir)  # E.g. directories given by "a/"
    histogram_distance_dict = {}
    # Iterate over each letter label
    for label_letter in label_letters:  # Iterate over each label letter categ.
        print("Label letter is: {}".format(label_letter))
        sub_dir = os.path.join(label_dir, label_letter)
        label_names = os.listdir(sub_dir)  # Lists labels as directories
        for label in label_names:  # Iterate over each individual label
            histogram_distance_dict[label] = np.nan
            # Get files in sub-sub directory
            label_dir_name = os.path.join(label_dir, label_letter, label)
            label_dir_files = os.listdir(label_dir_name)

            # Get input image files and number of images
            input_imgs = [file for file in label_dir_files if file.endswith(
                ".jpg")]
            N = len(input_imgs)
            # Hash table to efficiently check if we've seen images before
            seen = {}
            if N <= 1:
                continue
            histogram_distance_dict[label] = 0
            # Now iterate through all the files for each ground truth label
            if N > 0:
                for f1 in input_imgs:
                    for f2 in input_imgs:
                        if f1 != f2:  # Don't need to compare the same file
                            # Check if we've seen image before
                            if seen.get((f1, f2)) is not None or \
                                    seen.get((f2, f1)) is not None:
                                continue
                            # Build both histograms
                            hist1 = build_histogram(descriptor_list[f1], kmeans,
                                                    n_clusters)
                            hist2 = build_histogram(descriptor_list[f2], kmeans,
                                                    n_clusters)
                            # Add to label distribution dictionary with distance
                            histogram_distance_dict[label] += float(
                                get_difference_histograms(hist1, hist2,
                                                          metric=metric))

                            # Now we've seen the pair of images
                            seen[(f1, f2)] = 0

                #  Take the mean by dividing by all combinations of images
                histogram_distance_dict[label] /= (N ** 2 - N) / 2

    return histogram_distance_dict


def plot_eval_results(ks, distances, out_file_path="", metric="L2 Norm"):
    """Function for plotting average histogram distance between images with
    the same labels as a function of the number of k-means clusters used."""

    plt.plot(ks, distances)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Distance ({})".format(metric))
    plt.title("Average Histogram Distance Between "
              "Histograms as a Function of K-Means Clusters ")
    plt.savefig(out_file_path)
    print("Figure saved in: {}".format(out_file_path))


def create_feature_matrix(img_path, n_clusters=n_clusters):
    """Main function for creating a matrix of size N_images x n_clusters
    using SIFT and histogramming of the descriptors by a clustering
    algorithm."""
    # Make clustering algorithm
    kmeans = KMeans(n_clusters=n_clusters)
    img_files = os.listdir(img_path)  # img_file should be
    # "/home/yaatehr/datasets/seg_data/images/training/"
    # print(img_files)
    print(len(img_files))
    descriptor_path = "/home/yaatehr/programs/spatial_LDA/data" \
                      "/image_descriptors_dictionary_%s_keypoints.pkl" % \
                      n_keypoints
    print(descriptor_path)
    with open(descriptor_path, "rb") as f:
        descriptor_list_dic = pickle.load(f)
        # with open(descriptor_path,"rb") as f:
        print(descriptor_path)
        descriptor_list_dic = pickle.load(f)
    descriptor_list_dic = {} #f: descriptor vectors
    num_files = 0
    for l in img_files: 
        label_path = os.path.join(img_path, l) #a/
        labels = os.listdir(label_path) #a/amusement_park
        for label in labels:
            singular_label_path = os.path.join(label_path, label)
            print(singular_label_path)
            images = os.listdir(singular_label_path)
            for f in images:
                if f[-3:] != 'jpg':
                    continue
                num_files += 1
                if num_files %99==0:
                    print(str(num_files+1)+" files processed")
                A = cv.imread(os.path.join(singular_label_path, f)) # read
                _, des = get_feature_vector(A)
                descriptor_list_dic[f]= des
    # with open(descriptor_path, "wb") as f:
    #     pickle.dump(descriptor_list_dic, f)
    # print("Dumped descriptor dictionary of %s keypoints" %n_keypoints)
    vstack = np.vstack([i for i in list(descriptor_list_dic.values()) if
                        i is not None and i.shape[0] == n_keypoints])
    print(vstack.shape)
    kmeans.fit(vstack)
    kmeans_path = "/home/yaatehr/programs/spatial_LDA/data/kmeans_" \
                  "%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints)
    with open(kmeans_path, "wb") as f:
        pickle.dump(kmeans, f)
        # kmeans = pickle.load(f)
    print('dumped kmeans model')

    # Get image files
    M = []
    num_files = 0
    for l in img_files:
        label_path = os.path.join(img_path, l)  # a/
        labels = os.listdir(label_path)  # a/amusement_park
        for label in labels:
            singular_label_path = os.path.join(label_path, label)
            print(singular_label_path)
            images = os.listdir(singular_label_path)
            for f in images:  # Iterate over all image files
                if f[-3:] != "jpg":
                    continue
                if num_files % 100 == 0:
                    print(str(num_files) + " files processed")
                des = descriptor_list_dic[f]  # Get keypoints/descriptors from SIFT
                print(des)
                if des is None or des.shape[0] != n_keypoints:
                    print('hi')
                    continue
                histogram = build_histogram(des, kmeans, n_clusters)

                M.append(histogram)  # Append to output matrix
                num_files += 1
    return M, kmeans


def create_feature_matrix_cnn(img_path, model, n_clusters=n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    img_files = os.listdir(img_path)
    print(len(img_files))
    with open(
            "/home/yaatehr/programs/spatial_LDA/data/cnn_descriptors_dict01"
            ".pkl",
            "rb") as f:
        descriptor_list_dic = pickle.load(f)
    # print(img_files)
    # descriptor_list_dic = {}
    # for f in img_files:
    #    test_img = io.imread(f)
    #    inputs = resnet_transform(test_img)
    #    inputs = inputs.unsqueeze(0)
    #    des = model(inputs).view(4*49,128).detach().numpy()
    #    descriptor_list_dic[f]= des
    #    del test_img
    box_path = "/home/yaatehr/programs/spatial_LDA/data" \
               "/cnn_descriptors_dict01.pkl"
    local_path = "/Users/yaatehr/Programs/spatial_LDA/cnn_descriptors_dict01" \
                 ".pkl"
    with open(box_path, "wb") as f:
        pickle.dump(descriptor_list_dic, f)
    vstack = np.vstack([i for i in list(descriptor_list_dic.values()) if
                        i is not None and i.shape[0] == n_cnn_keypoints])
    print(vstack.shape)
    kmeans.fit(vstack)

    # Get image files
    M = []
    num_files = 0
    for f in img_files:  # Iterate over all image files
        if num_files % 100 == 0:
            print(str(num_files) + " files processed")
        des = descriptor_list_dic[f]  # Get keypoints/descriptors from CNN
        if des is None or des.shape[0] != n_cnn_keypoints:
            continue
        histogram = build_histogram(des, kmeans, n_clusters)

        M.append(histogram)  # Append to output matrix
        num_files += 1
    return M

def make_dataset_directory(dataset_filepath):
    BOX_DATA_ROOT = "/home/yaatehr/programs/datasets/seg_data/images/training"
    grayscaleDataset = ADE20K(grayscale=True, root=BOX_DATA_ROOT, transform = None, useStringLabels=True, randomSeed=49)
    # dataset = get_single_loader(grayscaleDataset, batch_size=1, shuffle_dataset=False)
    mostCommonLabels =  list(map(lambda x: x[0], grayscaleDataset.counter.most_common(25)))
    grayscaleDataset.selectSubset(mostCommonLabels, normalizeWeights=True)
    if not os.path.exists(dataset_filepath):
        os.mkdir(dataset_filepath)
        print("created directory")
    print('going into loop')
    for idx, (image, label) in enumerate(grayscaleDataset):
        # print('hi')
        letter = label[0]
        letter_path = os.path.join(dataset_filepath, letter)
        if not os.path.exists(letter_path):
            os.mkdir(letter_path)
        label_path = os.path.join(letter_path, label)
        if not os.path.exists(label_path):
            os.mkdir(label_path)
        img_path = grayscaleDataset.image_paths[idx]
        image_filename = img_path.split('/')[-1]
        complete_pathname = os.path.join(label_path, image_filename)
        print(complete_pathname)
        cv.imwrite(complete_pathname, image)


def main():
    dataset_path = "/home/yaatehr/programs/spatial_LDA/data/descriptors_test_1"
    # M = create_feature_matrix(dataset_npath)
    model = get_model()
    CnnMatrix = create_feature_matrix_cnn(dataset_path, model)

    with open("/home/yaatehr/programs/spatial_LDA/data/cnn_feature_matrix",
              "wb") as f:
        pickle.dump(CnnMatrix, f)


if __name__ == "__main__":
    main()
