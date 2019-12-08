import argparse
from feature_extraction import evaluate_kmeans, build_histogram
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def main_eval():
    # Parse arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--metric", \
    #                    help="Distance metric ('l2', 'l1', 'kl')")
    parser.add_argument("-n", "--num_clusters", \
                        help="Number of clusters", type=int)
    parser.add_argument("-k", "--num_keypoints", \
                        help="Number of keypoints", type=int)
    parsed_args = vars(parser.parse_args())

    # Get params
    num_clusters = parsed_args["num_clusters"]
    num_keypoints = parsed_args["num_keypoints"]

    # Get paths to filenames
    f_kmeans = "/home/yaatehr/programs/spatial_LDA/data/kmeans_%s_clusters_" \
               "%s_keypoints.pkl" % (num_clusters, num_keypoints)
    f_descriptor = "/home/yaatehr/programs/spatial_LDA/data" \
                   "/image_descriptors_dictionary_%s_keypoints.pkl" % \
                   num_keypoints

    for metric in ["l2", "l1", "kl"]:
        print("Metric is: {}, keypoints: {}, clusters: {}".format(metric,
                                                                  num_keypoints,
                                                                  num_clusters))
        # Load pickled files
        with open(f_kmeans, "rb") as f:
            kmeans = pickle.load(f)
        f.close()
        with open(f_descriptor, "rb") as f:
            descriptor_list = pickle.load(f)
        f.close()

        # Evaluate model with different params/hyperparams
        histogram_distance_dict = evaluate_kmeans(descriptor_list, kmeans,
                                                  num_clusters, metric=metric)
        print("Evaluation finished")
        # Pickle distance dictionary
        f_out_pickle = "/home/yaatehr/programs/spatial_LDA/data/EVAL_kmeans_" \
                       "%s_clusters_%s_keypoints_%s_metric.pkl" % (num_clusters,
                                                                   num_keypoints,
                                                                   metric)
        with open(f_out_pickle, "wb") as f:
            pickle.dump(histogram_distance_dict, f)
            f.close()
        print("Pickle file dumped at: {}".format(f_out_pickle))
    
    

def plot_histograms_for_labels(n_keypoints, n_clusters):
    label_path = "/home/yaatehr/programs/seg_data/images/dataset1/"
    letters = os.listdir(label_path)
    f_kmeans = "/home/yaatehr/programs/spatial_LDA/data/kmeans_%s_clusters_" \
               "%s_keypoints.pkl" % (n_clusters, n_keypoints)
    f_descriptor = "/home/yaatehr/programs/spatial_LDA/data" \
                   "/image_descriptors_dictionary_%s_keypoints.pkl" % \
                   n_keypoints
    with open(f_kmeans, 'rb') as f:
        kmeans = pickle.load(f)
    with open(f_descriptor, 'rb') as f:
        descriptor_list = pickle.load(f)
    for l in letters:
        label_path = os.path.join(label_path, l)
        labels = os.listdir(label_path)
        for label in labels:
            singular_label_path = os.path.join(label_path, label)
            plot_histograms_per_label(label_path, n_keypoints, kmeans, descriptor_list, 0.01)


def main_aggregate_pkl_files():
    print("HERE")
    kmeans_eval_dir = "/home/yaatehr/programs/spatial_LDA/data/"
    files = os.listdir(kmeans_eval_dir)
    kmeans_eval_files = [file for file in files if file.startswith("EVAL")]
    kmeans_eval_aggregate_dict = {}
    print(kmeans_eval_files)
    for kmeans_file in kmeans_eval_files:
        print("HERE1")
        split_fname = kmeans_file.split("_")
        num_clusters = split_fname[2]
        num_keypoints = split_fname[4]
        metric = split_fname[6]
        with open(os.path.join(kmeans_eval_dir, kmeans_file), "rb") as f:
            kmeans_eval_aggregate_dict[(num_clusters, num_keypoints, metric)]\
                = \
                pickle.load(f)
            f.close()

    # Dump pickle file information
    kmeans_aggregate_dict_file = \
        "/home/yaatehr/programs/spatial_LDA/data/kmeans_aggregate_eval_dict.pkl"
    with open(kmeans_aggregate_dict_file, "wb") as f:
        pickle.dump(kmeans_eval_aggregate_dict, f)
        f.close()

def plot_histograms_per_label(label_path,n_keypoints, kmeans, descriptor_list, percentage_plotted = 0.01):
    img_files = os.listdir(label_path)
    fig, ax = plt.subplots(1, 1)
    label = label_path.split("/")[-1]
    n_clusters = kmeans.cluster_centers_.shape[0]
    for f in img_files:
        if f[-3:] != 'jpg':
            continue
        if np.random.random() < percentage_plotted:
            #Plot image histogram
            des = descriptor_list[f]

            histogram = build_histogram(des, kmeans, kmeans.cluster_centers_.shape[0])
            plt.plot(histogram)
    plt.xlabel("features bag of words")
    plt.title("Histogram distribution for label %s" %label)
    plt.savefig("plots/histogram_distribution_label_%s_%s_keypoints_%s_clusters.png"%(label, n_keypoints, n_clusters))

def main_plot():

    # Helper function
    def compute_weighted_average():
        seg_data_path = "/home/yaatehr/programs/datasets/seg_data/images/training/"
        weight_dict = {}
        subfolders = os.listdir(seg_data_path)
        for subfolder in subfolders:
            label_classes = os.listdir(os.path.join(seg_data_path, subfolder))
            for label in label_classes:
                files = os.listdir(os.path.join(seg_data_path, subfolder, label))
                weight_dict[label] = len([file for file in files if
                                          file.endswith(".jpg")])
        return weight_dict

    def compute_average_dist(kmeans_eval):
        weight_dict = compute_weighted_average()
        N = np.sum(list(weight_dict.values()))
        
    kmeans_eval_file = "/home/yaatehr/programs/spatial_LDA/data/kmeans_aggregate_eval_dict.pkl"
    with open(kmeans_eval_file, "rb") as f:
        kmeans_eval_dict = pickle.load(f)
        f.close()
    keys = list(kmeans_eval_dict.keys())
    for key in keys:
        metric = key[2]
        print(metric)
        if metric == "l2":
            pass
        if metric == "l1":
            print(kmeans_eval_dict[key])
        if metric == "kl":
            print(kmeans_eval_dict[key])



if __name__ == "__main__":
    # main_eval()
    plot_histograms_for_labels(150, 150)
