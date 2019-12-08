"""This file contains code for a latent dirichlet allocation (SLDA),
which uses principles from maximum likelihood estimation and
Expectation-Maximization."""

# Native Python imports
import os
import copy

# External package imports
import numpy as np
import cv2 as cv
import pickle

# External package imports
import numpy as np
import cv2 as cv
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.cluster import KMeans
from scipy.special import kl_div
from dataset import *
# Custom module imports
#import dataset
import feature_extraction
import crop_images
from feature_extraction import n_keypoints, n_cnn_keypoints, n_clusters,\
 feature_model, cnn_num_layers_removed, num_most_common_labels_used

#n_keypoints = 100
# n_keypoints=49*4
n_topics = 20


class LDA2:
    """Class that implements Latent Dirichlet Allocation using an
    Expectation-Maximization framework.  This function iterates through the
    following two steps to find a locally-optimal maximum likelihood estimate of
    the relevant posterior distribution.

    Arguments:
        features: A numpy array of size (N_images x N_descriptors)
        alpha (float): A parameter for our LDA model (TODO: add on here).
        beta (float): A parameter for our LDA model (TODO: add on here).

    """
    # TODO: How do we represent the cardinality of our "vocabulary"
    def __init__(self, data_path, feature_path, alpha=1, beta=1, eps=1e-5,
                 n_topics=10, V=100):
        self.data_path = data_path  # File path for data
        self.feature_path = feature_path
        self.alpha = alpha  # Dirichlet dist hyperparameter
        self.beta = beta  # Dirichlet dist hyperparameter
        self.log_likelihood = None  # Array of log likelihoods
        self.parameters = None  # Numpy vector of parameters
        self.eps = eps  # Convergence threshold
        self.keypoints = None
        self.n_topics = n_topics
        self.get_data_matrix()  # Call in constructor method
        # self.m_documents = self.M.shape[0]
        self.vocab_size = V
        # self.init_LDA()  # Call in constructor method

    def get_data_matrix(self):
        if type(feature_path) == list:
            self.M = feature_path #LAZY OVERRIDE sorry lol
        else:
            with open(self.feature_path, 'rb') as f:
                self.M = pickle.load(f)
        self.m_documents = None

    def off_the_shelf_LDA(self):
        lda = LDA(n_components=self.n_topics)
        lda.fit(self.M)
        return lda

    def init_LDA(self):
        self.M_word_topic = np.zeros((self.k_topics, self.m_documents)) #(k x m)
        self.M_topic_vocab = np.zeros((self.k_topics, self.vocab_size)) #(k x v)
        self.M_count_k = np.zeros((self.k_topics, 1))  # (k x 1)

        # Now we need to randomly initialize documents and vocab over topics
        self.document_topics = np.random.randint(1, self.k_topics,
                                                 size=(self.k_topics,
                                                       self.m_documents))
        # TODO: FINISH THIS

    def compute_conditional_dist(self, m, n):
        # Return a probability distribution for word n in document m
        # belonging to a topic
        vector = [self.M_word_topic[k,m]]


    def sample_phi_from_dirichlet(self, n=1):
        """Function to sample from a Dirichlet distribution.

        Arguments:
            n (int): The number of samples from the Dirichlet distribution
                this function returns.

        Returns:
            A np array of samples from the Dirichlet distribution, of size n.
        """

        return np.random.dirichlet(self.beta, size=n)

    def gibbs_sampler(self, T):
        for t in T:  # Iterate over timesteps
            for i in []:
                pass


    def sample_pi_from_dirichlet(self, n=1):
        """Function to sample from a Dirichlet distribution.

        Arguments:
            n (int): The number of samples from the Dirichlet distribution
                this function returns.

        Returns:
            A np array of samples from the Dirichlet distribution, of size n.
        """

        return np.random.dirichlet(self.alpha, size=n)

    def find_params(self):
        """Function that iterates over the expectation and maximization steps
        until the parameter estimates converge to their optimal set."""

        old_params = np.zeros(self.parameters.shape)

        # Iterate until our parameters converge
        while np.linalg.norm(old_params - new_params) > self.eps:
            # Take expectation step
            self.expectation_step()

            # Take maximization step
            self.maximization_step()

            old_params = copy.deepcopy(self.parameters)

        return self.parameters

def compute_num_labels_in_cluster(cluster_predictions, actual_dic):
    """Given the cluster_predictions, that maps id:cluster, actual_dic that maps id:label
    and actual_labels which is all the labels in actual_dic, returns a count of label:count
    that is found in the cluster."""
    actual_labels = list(actual_dic.values())
    count = {i: 0 for i in actual_labels}
    for k in cluster_predictions:
        label = actual_dic[k]
        count[label] += 1
    return count

def compute_symmetric_KL(dist_a, dist_b):
    return 0.5 * np.sum((kl_div(dist_a, dist_b))) + 0.5 * np.sum((kl_div(dist_b, dist_a)))

def compute_probability_distr_difference(dist1, dist2):
    """computes l2 distance between two probabilty distributions"""
    return np.sum(np.square(dist1-dist2))

def evaluate_dataset_cnn():
    data_dir = getDirPrefix(num_most_common_labels_used, feature_model, cnn_num_layers_removed)
    dataset = ADE20K(root=getDataRoot(), transform=resnet_transform, useStringLabels=True, randomSeed=49)
    mostCommonLabels = list(map(lambda x: x[0], dataset.counter.most_common(num_most_common_labels_used)))
    dataset.selectSubset(mostCommonLabels, normalizeWeights=True)
    batch_size = 50
    loader = get_single_loader(dataset=dataset, batch_size=batch_size)

    actual_dic = {}
    with open(os.path.join(data_dir, "predicted_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "rb") as f:
        predicted = pickle.load(f)
    with open(os.path.join(data_dir, "clustered_images_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), 'rb') as f:
        clustered_images = pickle.load(f)
    with open(os.path.join(data_dir, "prob_distrs_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "rb") as f:
        prob_distrs = pickle.load(f)
    actual_dic = dataset.getImpathToLabelDict()
    num_in_each_cluster = {} #maps cluster to dictionary of label to count
    for cluster in clustered_images:
        #get number of labels in each cluster
        dic = compute_num_labels_in_cluster(clustered_images[cluster], actual_dic)
        num_in_each_cluster[cluster] = dic
    with open(os.path.join(data_dir, "num_in_each_cluster_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(num_in_each_cluster, f)   
    #get average l2 distance between pairs of each cluster
    avg_dist = {}  #maps cluster to average l2 distance
    avg_kl = {}  #maps cluster to average kl distance

    for label in dataset.class_indices.keys():
        labelIndices = dataset.class_indices[label]
        dist_count = 0
        kl_count = 0
        counter =  0
        for i in labelIndices:
            j = dataset.image_paths[i]
            if j not in prob_distrs:
                continue
            for w in labelIndices:
                k = dataset.image_paths[w]
                if j==k:
                    continue
                if k not in prob_distrs:
                    continue
                probj = prob_distrs[j]
                probk = prob_distrs[k]
                kl = compute_symmetric_KL(probj, probk)
                kl_count += kl
                dist = compute_probability_distr_difference(probj, probk)
                dist_count += dist
                counter += 1
        avg_dist[label] = dist_count/counter if counter!=0 else None
        avg_kl[label] = kl_count/counter if counter!=0 else None

    with open(os.path.join(data_dir, "avg_dist_in_label_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(avg_dist, f)
    with open(os.path.join(data_dir, "avg_kl_in_label_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(avg_kl, f)


def evaluate_dataset_sift():
    data_dir = getDirPrefix(num_most_common_labels_used, "sift")
    dataset = ADE20K(root=getDataRoot(), transform=resnet_transform, useStringLabels=True, randomSeed=49)
    mostCommonLabels =  list(map(lambda x: x[0], dataset.counter.most_common(num_most_common_labels_used)))
    dataset.selectSubset(mostCommonLabels, normalizeWeights=True)
    sift_feature_path = save_root + "feature_matrix_%s_keypoints_%s_clusters" %(n_keypoints, n_clusters)

    with open(sift_feature_path, "rb") as f:
        feature_tup = pickle.load(f)

    hist_list, index_mask = feature_tup
    dataset.applyMask(index_mask)

    actual_dic = {}
    with open(os.path.join(data_dir, "predicted_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "rb") as f:
        predicted = pickle.load(f)
    with open(os.path.join(data_dir, "clustered_images_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), 'rb') as f:
        clustered_images = pickle.load(f)
    with open(os.path.join(data_dir, "prob_distrs_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "rb") as f:
        prob_distrs = pickle.load(f)
    actual_dic = dataset.getImpathToLabelDict()
    num_in_each_cluster = {} #maps cluster to dictionary of label to count
    for cluster in clustered_images:
        #get number of labels in each cluster
        dic = compute_num_labels_in_cluster(clustered_images[cluster], actual_dic)
        num_in_each_cluster[cluster] = dic
    with open(os.path.join(data_dir, "num_in_each_cluster_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(num_in_each_cluster, f)   
    #get average l2 distance between pairs of each cluster
    avg_dist = {}  #maps cluster to average l2 distance
    avg_kl = {}  #maps cluster to average kl distance

    for label in dataset.class_indices.keys():
        labelIndices = dataset.class_indices[label]
        dist_count = 0
        kl_count = 0
        counter =  0
        for i in labelIndices:
            j = dataset.image_paths[i]
            if j not in prob_distrs:
                continue
            for w in labelIndices:
                k = dataset.image_paths[w]
                if j==k:
                    continue
                if k not in prob_distrs:
                    continue
                probj = prob_distrs[j]
                probk = prob_distrs[k]
                kl = compute_symmetric_KL(probj, probk)
                kl_count += kl
                dist = compute_probability_distr_difference(probj, probk)
                dist_count += dist
                counter += 1
        avg_dist[label] = dist_count/counter if counter!=0 else None
        avg_kl[label] = kl_count/counter if counter!=0 else None

    with open(os.path.join(data_dir, "avg_dist_in_label_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(avg_dist, f)
    with open(os.path.join(data_dir, "avg_kl_in_label_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(avg_kl, f)

def evaluate_main(cnn_mode = False):
    # labels = ["06c54", "011k07", "099ssp"] #labels in descriptors_test_1
    m_dir = "/home/yaatehr/programs/datasets/seg_data/images/dataset1/" #labels
    data_dir = '/home/yaatehr/programs/spatial_LDA/data/'
    if cnn_mode:
        data_dir = getDirPrefix()
    actual_dic = {}
    with open(os.path.join(data_dir, "predicted_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "rb") as f:
        predicted = pickle.load(f)
    with open(os.path.join(data_dir, "clustered_images_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), 'rb') as f:
        clustered_images = pickle.load(f)
    with open(os.path.join(data_dir, "prob_distrs_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "rb") as f:
        prob_distrs = pickle.load(f)
    img_files = os.listdir(m_dir)
    for l in img_files: 
        label_path = os.path.join(m_dir, l) #a/
        labels = os.listdir(label_path) #a/amusement_park
        for label in labels:
            singular_label_path = os.path.join(label_path, label)
            print(singular_label_path)
            dic = crop_images.map_image_id_to_label(singular_label_path, label)
            actual_dic.update(dic)
    num_in_each_cluster = {} #maps cluster to dictionary of label to count
    for cluster in clustered_images:
        #get number of labels in each cluster
        dic = compute_num_labels_in_cluster(clustered_images[cluster], actual_dic)
        num_in_each_cluster[cluster] = dic
    with open(os.path.join(data_dir, "num_in_each_cluster_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(num_in_each_cluster, f)

    #get average l2 distance between pairs of each cluster
    avg_dist = {}  #maps cluster to average l2 distance
    avg_kl = {}  #maps cluster to average kl distance
    for l in img_files: #should be in label
        label_path = os.path.join(m_dir, l)
        labels = os.listdir(label_path)
        for label in labels:
            singular_label_path = os.path.join(label_path, label)
            print(singular_label_path)
            images = os.listdir(singular_label_path)
            dist_count = 0
            kl_count = 0
            counter =  0
            for j in images:
                if j[-3:] != 'jpg':
                    continue
                if j not in prob_distrs: #not enough keypoints
                    continue
                for k in images:
                    if j==k:
                        continue
                    if k[-3:] != 'jpg':
                        continue
                    if k not in prob_distrs:
                        continue
                    probj = prob_distrs[j]
                    probk = prob_distrs[k]
                    kl = compute_symmetric_KL(probj, probk)
                    kl_count += kl
                    dist = compute_probability_distr_difference(probj, probk)
                    dist_count += dist
                    counter += 1
            avg_dist[label] = dist_count/counter if counter!=0 else None
            avg_kl[label] = kl_count/counter if counter!=0 else None
    with open(os.path.join(data_dir, "avg_dist_in_label_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(avg_dist, f)
    with open(os.path.join(data_dir, "avg_kl_in_label_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(avg_kl, f)


def main():
    #TODO: FILL IN feature_path
    dataset_path = "/home/yaatehr/programs/datasets/seg_data/images/dataset1/"
    # feature_extraction.make_dataset_directory(dataset_path)
    sift_feature_path = "/home/yaatehr/programs/spatial_LDA/data/sift_feature_matrix_%s_keypoints_%s_clusters" %(n_keypoints, n_clusters)
    M, kmeans = feature_extraction.create_feature_matrix(dataset_path)
    with open(sift_feature_path, "wb") as f:
        print(sift_feature_path)
        pickle.dump(M, f)
    print("dumped feature matrix")

    with open(sift_feature_path, "rb") as f:
        M = pickle.load(f)
    #CnnM = feature_extraction.create_feature_matrix_cnn(dataset_path)
    # feature_path = "/home/yaatehr/programs/spatial_LDA/data/features1.pkl"
    # feature_path = "/home/yaatehr/programs/spatial_LDA/data/cnn_feature_matrix"
    # with open(feature_path, "rb") as f:
        # CnnM = pickle.load(f)
    lda = LDA2("", sift_feature_path, n_topics = n_topics)  # Make the class
    lda_model = lda.off_the_shelf_LDA()  # Fit the sklearn LDA model
    predicted = {}
    img_files = os.listdir(dataset_path)
    descriptor_path = "/home/yaatehr/programs/spatial_LDA/data" \
                      "/image_descriptors_dictionary_%s_keypoints.pkl" % \
                      n_keypoints
    # with open ("/home/yaatehr/programs/spatial_LDA/data/img_descriptors_dic1.pkl", "rb") as f:
    with open (descriptor_path, "rb") as f:
        descriptor_dic = pickle.load(f)
    predicted_cluster = {} #dictionary of imgid: cluster
    cluster_dic = {} #ictionary of cluster: [images in cluster]
    prob_distr_dic = {} #maps id: probability distribution over clusters
    kmeans_path = "/home/yaatehr/programs/spatial_LDA/data/kmeans_%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints)

    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)
    num_files = 0
    for l in img_files: 
        label_path = os.path.join(dataset_path, l) #a/
        labels = os.listdir(label_path) #a/amusement_park
        for label in labels:
            singular_label_path = os.path.join(label_path, label)
            print(singular_label_path)
            images = os.listdir(singular_label_path)
            for f in images:
                if f[-3:] != 'jpg':
                    continue
                if num_files % 100 == 0:
                    print(num_files)
                des = descriptor_dic[f]
                if des is None or des.shape[0] != n_keypoints: #only use images with n_keypoints
                    continue
                feature = feature_extraction.build_histogram(des, kmeans, n_clusters)
                predictions = lda_model.transform(np.reshape(feature, (1, feature.size)))
                prob_distr_dic[f] = predictions
                predicted_class = np.argmax(predictions, axis=1)[0]
                predicted_cluster[f] = predicted_class
                if predicted_class in cluster_dic:
                    cluster_dic[predicted_class].append(f)
                else:
                    cluster_dic[predicted_class] = [f]
                num_files += 1
    with open ("/home/yaatehr/programs/spatial_LDA/data/predicted_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters), "wb") as f:
        pickle.dump(predicted_cluster, f)
    with open("/home/yaatehr/programs/spatial_LDA/data/clustered_images_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters), "wb") as f:
        pickle.dump(cluster_dic, f)
    with open("/home/yaatehr/programs/spatial_LDA/data/prob_distrs_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters), "wb") as f:
        pickle.dump(prob_distr_dic, f)   
    # Now we can predict!

def build_cnn_predictions():
    """NOTE this is using the dataloader, not porting the directory structure over. 
    Hopefully this will be useful if we need to change the dataset parameters."""
    cnn_root = getDirPrefix(num_most_common_labels_used, feature_model, cnn_num_layers_removed)
    cnn_feature_path = os.path.join(cnn_root, "feature_matrix_%s_keypoints_%s_clusters" %(n_keypoints, n_clusters))
    if not os.path.exists(cnn_feature_path):
        hist_list, kmeans = feature_extraction.create_feature_matrix_cnn()
        with open(cnn_feature_path, "wb") as f:
            print(cnn_feature_path)
            pickle.dump(hist_list, f)
        print("dumped feature matrix")

    with open(cnn_feature_path, "rb") as f:
        hist_list = pickle.load(f)

    lda = LDA2("", cnn_feature_path, n_topics = n_topics)  # Make the class
    lda_model = lda.off_the_shelf_LDA()  # Fit the sklearn LDA model
    predicted = {}
    descriptor_path = os.path.join(cnn_root, \
                      "image_descriptors_dictionary_%s_keypoints.pkl" % \
                      n_keypoints)
    with open (descriptor_path, "rb") as f:
        descriptor_dic = pickle.load(f)
    predicted_cluster = {} #dictionary of imgid: cluster
    cluster_dic = {} #ictionary of cluster: [images in cluster]
    prob_distr_dic = {} #maps id: probability distribution over clusters

    kmeans_path = os.path.join(cnn_root, "kmeans_%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints))
    if not os.path.exists(kmeans_path) :
        kmeans_path = os.path.join(cnn_root, "batch_kmeans_%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints))

    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    dataset = ADE20K(root=getDataRoot(), transform=resnet_transform, useStringLabels=True, randomSeed=49)
    mostCommonLabels =  list(map(lambda x: x[0], dataset.counter.most_common(num_most_common_labels_used)))
    dataset.selectSubset(mostCommonLabels, normalizeWeights=True)
    batch_size = 50
    loader = get_single_loader(dataset=dataset, batch_size=batch_size)
    assert len(hist_list) == len(dataset)

    num_files = 0
    for i in range(len(hist_list)): 
        if num_files % 100 == 0:
            print(num_files)

        f= dataset.image_paths[i]
        feature = hist_list[i]
        predictions = lda_model.transform(np.reshape(feature, (1, feature.size)))
        prob_distr_dic[f] = predictions
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_cluster[f] = predicted_class
        if predicted_class in cluster_dic:
            cluster_dic[predicted_class].append(f)
        else:
            cluster_dic[predicted_class] = [f]
        num_files += 1
    with open (os.path.join(cnn_root, "predicted_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(predicted_cluster, f)
    with open(os.path.join(cnn_root, "clustered_images_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(cluster_dic, f)
    with open(os.path.join(cnn_root, "prob_distrs_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(prob_distr_dic, f)   
    # Now we can predict!



def build_sift_predictions():
    """NOTE this is using the dataloader, not porting the directory structure over. 
    Hopefully this will be useful if we need to change the dataset parameters."""
    save_root = getDirPrefix(num_most_common_labels_used, "sift", makedirs=True)
    sift_feature_path = os.path.join(save_root, "feature_matrix_%s_keypoints_%s_clusters" %(n_keypoints, n_clusters))

    if not os.path.exists(sift_feature_path):
        feature_tup, kmeans = feature_extraction.create_feature_matrix_sift()
        with open(sift_feature_path, "wb") as f:
            print(sift_feature_path)
            pickle.dump(feature_tup, f)
        print("dumped feature matrix")

    with open(sift_feature_path, "rb") as f:
        feature_tup = pickle.load(f)

    hist_list, index_mask = feature_tup

    lda = LDA2("", hist_list, n_topics = n_topics)  # Make the class
    lda_model = lda.off_the_shelf_LDA()  # Fit the sklearn LDA model
    predicted = {}
    descriptor_path = os.path.join(save_root,
                      "image_descriptors_dictionary_%s_keypoints.pkl" % \
                      n_keypoints)
    with open (descriptor_path, "rb") as f:
        descriptor_dic = pickle.load(f)
    predicted_cluster = {} #dictionary of imgid: cluster
    cluster_dic = {} #ictionary of cluster: [images in cluster]
    prob_distr_dic = {} #maps id: probability distribution over clusters

    kmeans_path = os.path.join(save_root, "kmeans_%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints))

    with open(kmeans_path, "rb") as f:
        kmeans = pickle.load(f)

    dataset = ADE20K(root=getDataRoot(), transform=None, useStringLabels=True, randomSeed=49)
    mostCommonLabels =  list(map(lambda x: x[0], dataset.counter.most_common(num_most_common_labels_used)))
    dataset.selectSubset(mostCommonLabels, normalizeWeights=True)
    dataset.applyMask(index_mask)

    num_files = 0
    for i in range(len(dataset)): 
        if num_files % 100 == 0:
            print(num_files)

        f = dataset.image_paths[i]
        feature = hist_list[i]
        predictions = lda_model.transform(np.reshape(feature, (1, feature.size)))
        prob_distr_dic[f] = predictions
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_cluster[f] = predicted_class
        if predicted_class in cluster_dic:
            cluster_dic[predicted_class].append(f)
        else:
            cluster_dic[predicted_class] = [f]
        num_files += 1
    with open (os.path.join(save_root, "predicted_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(predicted_cluster, f)
    with open(os.path.join(save_root, "clustered_images_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters), "wb")) as f:
        pickle.dump(cluster_dic, f)
    with open(os.path.join(save_root, "prob_distrs_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(prob_distr_dic, f)   
    # Now we can predict!

def ryan_test():
    dataset_path = "/home/rmsander/Documents/6.867/test_dir/WelshCorgi.jpeg"
    img = cv.imread(dataset_path)
    M = feature_extraction.get_feature_vector(img)
    print(M)

if __name__ == "__main__":
    # main()
    build_cnn_predictions()
    build_sift_predictions()
    evaluate_dataset_sift()
    evaluate_dataset_cnn()
