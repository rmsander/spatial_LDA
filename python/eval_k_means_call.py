import argparse
from feature_extraction import evaluate_kmeans
import pickle


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metric", \
                        help="Distance metric ('l2', 'l1', 'kl')")
    parser.add_argument("-n", "--num_clusters", \
                        help="Number of clusters")
    parser.add_argument("-k", "--num_keypoints", \
                        help="Number of keypoints")
    parsed_args = vars(parser.parse_args())

    # Get params
    metric = parsed_args["metric"]
    num_clusters = parsed_args["num_clusters"]
    num_keypoints = parsed_args["num_keypoints"]

    # Get paths to filenames
    f_kmeans = "/home/yaatehr/programs/spatial_LDA/data/kmeans_%s_clusters_" \
               "%s_keypoints.pkl" % (num_clusters, num_keypoints)
    f_descriptor = "/home/yaatehr/programs/spatial_LDA/data" \
                   "/image_descriptors_dictionary_%s_keypoints.pkl" % \
                   num_keypoints

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
                   "%s_clusters_" \
               "%s_keypoints.pkl" % (num_clusters, num_keypoints)
    with open(f_out, "rb") as f:
        pickle.dump(histogram_distance_dict, f)
        f.close()
    print("Pickle file dumped at: {}".format(f_out_pickle))


if __name__ == "__main__":
    main()
