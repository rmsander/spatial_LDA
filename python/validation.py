import numpy as np
from feature_extraction import get_feature_vector, build_histogram
from lda import LDA2
import pickle
from dataset import get_single_loader, ADE20K
import os
from skimage import io
import cv2 as cv
import crop_images
from lda import compute_num_labels_in_cluster, compute_symmetric_KL,compute_probability_distr_difference

n_clusters = 150
n_topics = 20
n_keypoints = 150


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

def evaluate_main():
    m_dir = "/home/yaatehr/programs/datasets/seg_data/images/dataset1_val/" #labels
    data_dir = '/home/yaatehr/programs/spatial_LDA/data/'
    actual_dic = {}
    with open(os.path.join(data_dir, "VAL_predicted_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "rb") as f:
        predicted = pickle.load(f)
    with open(os.path.join(data_dir, "VAL_clustered_images_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), 'rb') as f:
        clustered_images = pickle.load(f)
    with open(os.path.join(data_dir, "VAL_prob_distrs_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "rb") as f:
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
    with open(os.path.join(data_dir, "VAL_num_in_each_cluster_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
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
    with open(os.path.join(data_dir, "VAL_avg_dist_in_label_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(avg_dist, f)
    with open(os.path.join(data_dir, "VAL_avg_kl_in_label_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters)), "wb") as f:
        pickle.dump(avg_kl, f)


def main():
    dataset_path = "/home/yaatehr/programs/datasets/seg_data/images/dataset1_val/"
    # make_directory_for_validation(dataset_path)
    sift_feature_path = "/home/yaatehr/programs/spatial_LDA/data/sift_feature_matrix_%s_keypoints_%s_clusters" %(n_keypoints, n_clusters)
    kmeans_path = "/home/yaatehr/programs/spatial_LDA/data/kmeans_%s_clusters_%s_keypoints.pkl" % (n_clusters, n_keypoints)
    validation_img_path = ""
    with open(sift_feature_path, 'rb') as f:
        M = pickle.load(f)
    with open(kmeans_path, 'rb') as f:
        kmeans_model = pickle.load(f)
    lda = LDA2("", sift_feature_path, n_topics = n_topics)
    lda_model = lda.off_the_shelf_LDA()
    prob_distrs_validation = {} #id: prob distr over clusters
    cluster_dic_validation = {} #cluster:[images in cluster]
    predicted_cluster_dic = {} #imgid: cluster
    letter_files = os.listdir(dataset_path)
    num_files = 0
    for letter in letter_files:
        label_path = os.path.join(dataset_path, letter)
        labels = os.listdir(label_path)
        for label in labels:
            singular_label_path = os.path.join(label_path, label)
            print("creating predictions for : ", singular_label_path)
            images = os.listdir(singular_label_path) 
            for f in images:
                if f[-3:] == 'jpg':
                    continue
                if num_files%100 == 0:
                    print(num_files)
                img = cv.imread(f)
                prediction = get_prediction_for_image(img, lda_model, kmeans_model)
                prob_distrs_validation[f] = prediction
                predicted_class = np.argmax(prediction, axis=1)[0]
                if predicted_class in cluster_dic_validation:
                    cluster_dic_validation[predicted_class].append(f)
                else:
                    cluster_dic_validation[predicted_class] = [f]
                num_files += 1
    with open ("/home/yaatehr/programs/spatial_LDA/data/VAL_predicted_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters), "wb") as f:
        pickle.dump(predicted_cluster_dic, f)
    with open("/home/yaatehr/programs/spatial_LDA/data/VAL_clustered_images_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters), "wb") as f:
        pickle.dump(cluster_dic_validation, f)
    with open("/home/yaatehr/programs/spatial_LDA/data/VAL_prob_distrs_%s_topics_%s_keypoints_%s_clusters.pkl" %(n_topics, n_keypoints, n_clusters), "wb") as f:
        pickle.dump(prob_distrs_validation, f)  


if __name__ == "__main__":
    main()
    evaluate_main()
