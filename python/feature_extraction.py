import cv2 as cv
import numpy as np
import os
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_multilabel_classification
import scipy
from skimage import io
from scipy.special import kl_div as KL
import pickle
from torchvision import transforms
from skimage import io
import matplotlib.pyplot as plt
import argparse
from dataset import *
import torch
from tqdm import tqdm
import gc
import copy
from pca import featureNormalize

n_keypoints = 300  # hyperparameter, need to tune
n_cnn_keypoints = 4 * 49
n_clusters = 300  # also need to tune this
feature_model = "googlenetg" # see temp.txt for possible list of models
cnn_num_layers_removed = 3 # NOTE set to None for sift
num_most_common_labels_used = 25

def get_model():
    model = torch.hub.load('pytorch/vision', feature_model[:-1], pretrained=True)
    # cut off the last layer of this classifier
    new_classifier = torch.nn.Sequential(*list(model.children())[:-cnn_num_layers_removed])
    # print(new_classifier)
    model = new_classifier
    return model

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
    label_dir = "/home/yaatehr/programs/datasets/seg_data/images/dataset1/"
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

# Helper functions for segmentation code
def make_ID_mapping():
    segmented_counts_path = os.path.join("..", "data", "SEG_COUNTS.pkl")

    # Read pickle file of counts for each segmented image
    with open(segmented_counts_path, "rb") as f:
        segmentation_counts = pickle.load(f)
        f.close()

    # Get list of all unique IDs
    colorIDs = set()

    # Now get the number of classes for segmentation counts
    letters = list(segmentation_counts.keys())
    for letter in letters:
        for file in list(segmentation_counts[letter].keys()):
            unique_colors = list(segmentation_counts[letter][file])
            for color in unique_colors:
                colorIDs.add(color)

    # Create mapping
    color_copies = list(copy.deepcopy(colorIDs))
    print(
        "THERE ARE {} different color IDs in ADE 20k".format(len(color_copies)))
    colorID_map = {color_copies[i]: i for i in range(len(color_copies))}
    #print(colorID_map)

    # Now pickle IDs and mapping
    fname_IDS = os.path.join("..", "data", "color_IDs.pkl")
    fname_mapping = os.path.join("..", "data", "color_ID_Mapping.pkl")

    with open(fname_IDS, "wb") as ID_file:
        pickle.dump(colorIDs, ID_file)
        ID_file.close()

    with open(fname_mapping, "wb") as map_file:
        pickle.dump(colorIDs, map_file)
        map_file.close()

def eval_lda_segmented_labels(n_topics=20, n_keypoints=300,
                              n_clusters=300):

    # Now we want to compute the distribution of ground truth labels for each
    # latent topic
    with open(os.path.join("data","top25_sift",
                           "prob_distrs_%s_topics_%s_keypoints_%s_clusters.pkl" %(
                                   n_topics, n_keypoints, n_clusters)), "wb") as f:
        probability_distribution_dict = pickle.load(f)
        f.close()

    # Load segmentation pickle file
    fname_IDS = os.path.join("..", "data", "color_IDs.pkl")
    fname_mapping = os.path.join("..", "data", "color_ID_Mapping.pkl")
    fname_seg_counts = os.path.join("..", "data", "SEG_COUNTS.pkl")

    with open(fname_seg_counts, "rb") as f:
        seg_counts = pickle.load(f)
        f.close()

    with open(fname_IDS, "rb") as f:
        IDs = pickle.load(f)
        f.close()

    with open(fname_mapping, "rb") as f:
        mapping = pickle.load(f)
        f.close()

    prob_tensor = np.zeros((num_topics, len(IDs)))
    letters = list(seg_counts.keys())
    for letter in letters:
        for file in list(seg_counts[letter].keys()):
            color_map = np.array(list(mapping[seg_counts[letter][
                file]])).reshape((1,len(IDs)))
            print("COLOR MAP IS: {}".format(color_map))
            topic_dist = np.array(probability_distribution_dict[
                                      file]).reshape((len(num_topics),1))
            prob_tensor += topic_dist@color_map

    # Now pickle probability tensor/matrix
    fname_out =  os.path.join("..", "data", "topic_GT_label_dist_{"
                                            "}_topics_{}_keypoints_{"
                                            "}_clusters.pkl".format(n_topics,
                                                                    n_keypoints, n_clusters))

    with open(fname_out, "wb") as f:
        pickle.dump(prob_tensor, f)
        f.close()

    print("FINISHED PICKLING")




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
    #uncomment for pickled descriptor_list_dic
    # with open(descriptor_path, "rb") as f:
        # descriptor_list_dic = pickle.load(f)

    #uncomment to create descriptor_list_dic
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
    with open(descriptor_path, "wb") as f:
        pickle.dump(descriptor_list_dic, f)
    print("Dumped descriptor dictionary of %s keypoints" %n_keypoints)


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
                if des is None or des.shape[0] != n_keypoints:
                    continue
                histogram = build_histogram(des, kmeans, n_clusters)

                M.append(histogram)  # Append to output matrix
                num_files += 1
    return M, kmeans


def create_feature_matrix_cnn():
    # save_root = os.path.join(os.path.dirname(__file__), '../data')
    save_root = getDirPrefix(num_most_common_labels_used, feature_model, cnn_num_layers_removed=cnn_num_layers_removed)

    #DUMP DESCRIPTOR LIST
    descriptor_path = os.path.join(save_root,  "image_descriptors_dictionary_%s_keypoints.pkl" % \
                      (n_keypoints))

    kmeans_path = os.path.join(save_root, "kmeans_%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints))
    if not os.path.exists(kmeans_path) :
        kmeans_path = os.path.join(save_root, "batch_kmeans_%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints))

    if not (os.path.exists(kmeans_path) and os.path.exists(descriptor_path)):
        print("NO PATHS FOUND, overwriting descriptors and kmeans for: \n %s \n %s_clusters_%s_keypoints" % (save_root, n_clusters, n_keypoints))
        minibatchkmeans = MiniBatchKMeans(n_clusters=n_clusters)
        kmeans = KMeans(n_clusters)
        usingMinibatch = False
        model = get_model()
        transform = get_model_transform(feature_model)
        dataset = ADE20K(root=getDataRoot(), transform=transform, useStringLabels=True, randomSeed=49)
        mostCommonLabels =  list(map(lambda x: x[0], dataset.counter.most_common(num_most_common_labels_used)))
        dataset.selectSubset(mostCommonLabels, normalizeWeights=True)
        dataset.useOneHotLabels()
        num_images = len(dataset)
        batch_size = 79
        loader = get_single_loader(dataset=dataset, batch_size=batch_size)
        descriptor_dict = {}
        bar = tqdm(total= num_images)

        for step, (img,label) in enumerate(loader):
            outputs = model(img)
            unrolled_outputs = torch.flatten(outputs, start_dim=2).detach().numpy()
            # unrolled_outputs = np.apply_along_axis(featureNormalize, 1, unrolled_outputs)

            #build the descriptor map
            offset = step*batch_size

            descriptors = dict(zip(dataset.image_paths[offset:offset+batch_size], unrolled_outputs.tolist()))
            descriptor_dict.update(descriptors)
            batch_outputs_for_kmeans = unrolled_outputs.reshape(-1,unrolled_outputs.shape[-1])
            # minibatchkmeans.partial_fit(batch_outputs_for_kmeans)

            #build kmeans  fit input
            if step == 0:
                vstack = batch_outputs_for_kmeans
                continue
            vstack = np.vstack((vstack, batch_outputs_for_kmeans))
            del outputs
            gc.collect()
            bar.update(batch_size)

        bar.close()

        with open(descriptor_path, "wb") as f:
            pickle.dump(descriptor_dict, f)
            # kmeans = pickle.load(f)
        print('dumped descriptor dict for %s, %d, %s' % (feature_model, cnn_num_layers_removed, n_keypoints))

        # try:
        #     print("fitting generic kmeans")
        #     kmeans.fit(vstack)
        # except Exception as e:
        gc.collect()
        print("falling back to minibatch")
        kmeans = minibatchkmeans.fit(vstack)
        usingMinibatch = True

        # DUMP KMEANS
        if not usingMinibatch:
            kmeans_path = os.path.join(save_root, "kmeans_" \
                        "%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints))
        else:
            kmeans_path = os.path.join(save_root, "batch_kmeans_" \
                "%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints))

        with open(kmeans_path, "wb") as f:
            pickle.dump(kmeans, f)
            # kmeans = pickle.load(f)
        print('dumped kmeans model')
    else:
        print("LOADING CHECKPOINTS")
        with open(kmeans_path, 'rb') as f:
            kmeans = pickle.load(f)
        with open(descriptor_path, 'rb') as f:
            descriptor_dict = pickle.load(f)

    # build histograms for CNN Features
    hist_list = []
    print("building historgram")
    for path in descriptor_dict.keys():
        des = descriptor_dict[path]
        histogram = build_histogram(des, kmeans, n_clusters)
        hist_list.append(histogram)

    return hist_list, kmeans

def create_feature_matrix_sift():
    # save_root = os.path.join(os.path.dirname(__file__), '../data')
    save_root =  getDirPrefix(num_most_common_labels_used, "sift")

    #DUMP DESCRIPTOR LIST
    descriptor_path = os.path.join(save_root, "image_descriptors_dictionary_%s_keypoints.pkl" % \
                      (n_keypoints))

    kmeans_path = os.path.join(save_root, "kmeans_%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints))

    if not (os.path.exists(kmeans_path) and os.path.exists(descriptor_path)):
        print("NO PATHS FOUND, overwriting descriptors and kmeans for: \n %s \n %s_clusters_%s_keypoints" % (save_root, n_clusters, n_keypoints))
        minibatchkmeans = MiniBatchKMeans(n_clusters=n_clusters)
        kmeans = KMeans(n_clusters)
        dataset = ADE20K(root=getDataRoot(), transform=None, useStringLabels=True, randomSeed=49)
        mostCommonLabels =  list(map(lambda x: x[0], dataset.counter.most_common(num_most_common_labels_used)))
        dataset.selectSubset(mostCommonLabels, normalizeWeights=True)
        num_images = len(dataset)
        descriptor_dict = {}
        bar = tqdm(total= num_images)

        for step, (img,label) in enumerate(dataset):
            _, des = get_feature_vector(img)
            f = dataset.image_paths[step]
            descriptor_dict[f]= des
            if step%50 == 0:
                bar.update(50)

        bar.close()

        with open(descriptor_path, "wb") as f:
            pickle.dump(descriptor_dict, f)
        print("Dumped descriptor dictionary of %s keypoints" %n_keypoints)
        vstack = np.vstack([i for i in list(descriptor_dict.values()) if
                            i is not None and i.shape[0] == n_keypoints])
        print(vstack.shape)
        kmeans.fit(vstack)
        with open(kmeans_path, "wb") as f:
            pickle.dump(kmeans, f)
            # kmeans = pickle.load(f)
        print('dumped kmeans model')

        hist_list = []
        index_mask = []
        print("building historgram")
        for i, path in enumerate(dataset.image_paths):
            des = descriptor_dict[path]

            if des is None or des.shape[0] != n_keypoints:
                index_mask.append(False)
                continue
            histogram = build_histogram(des, kmeans, n_clusters)
            hist_list.append(histogram)
            index_mask.append(True)
    else:
        with open(kmeans_path, 'rb') as f:
            kmeans = pickle.load(f)
        with open(descriptor_path, 'rb') as f:
            descriptor_dic = pickle.load(f)
        dataset = ADE20K(root=getDataRoot(), transform=None, useStringLabels=True, randomSeed=49)
        mostCommonLabels =  list(map(lambda x: x[0], dataset.counter.most_common(num_most_common_labels_used)))
        dataset.selectSubset(mostCommonLabels, normalizeWeights=True)
        hist_list = []
        index_mask = []
        print("building historgram")
        for i, path in enumerate(dataset.image_paths):
            des = descriptor_dic[path.split('/')[-1]]

            if des is None or des.shape[0] != n_keypoints:
                index_mask.append(False)
                continue
            histogram = build_histogram(des, kmeans, n_clusters)
            hist_list.append(histogram)
            index_mask.append(True)

    return (hist_list, index_mask), kmeans



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
        io.imsave(complete_pathname, image)


def main():
    dataset_path = "/home/yaatehr/programs/spatial_LDA/data/descriptors_test_1"
    # M = create_feature_matrix(dataset_npath)
    # model = get_model()
    # # CnnMatrix = create_feature_matrix_cnn(dataset_path, model)

    # with open("/home/yaatehr/programs/spatial_LDA/data/cnn_feature_matrix",
    #           "wb") as f:
    #     pickle.dump(CnnMatrix, f)
    eval_dir = os.path.join("..", "data", "top25_sift")
    files = os.listdir(eval_dir)
    files_to_use = [file for file in files if file.beginswith("prob_distrs")]
    print("FILES: {}".format(files_to_use)
    for file in files_to_use:
        fname_split = file.split("_")
        topics, keypoints, clusters = fname_split[2], fname_split[4], \
                                      fname_split[6]
        print(topics, keypoints, clusters)
        eval_lda_segmented_labels(topics, keypoints, clusters)

if __name__ == "__main__":
    main()
