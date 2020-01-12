# Spatial Latent Dirichlet Allocation
## Overview
This repository contains the implementation of an unsupervised image organization model for use in semi-automation of the image annotation and data curation process.  It uses Latent Dirichlet Allocation (LDA), Scale-Invariant Feature Transform (SIFT), and pre-trained Convolutional Neural Network (CNN) filters to cluster unlabeled images into different categories using latent features.

## Block Diagram
![block_diagram](https://github.com/rmsander/spatial_LDA/blob/master/images_readme/pipeline.png)

## Installation
To install the requirements for this codebase, you can use either `conda` or `pip`.  In the root directory of this codebase (`spatial_LDA/`):
* `conda env create -f environment.yml` (**conda**)
* `pip install -r requirements.txt` (**pip**)

## Dataset
For this project, we used the [ADE20K dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/), which consists of RGB and segmented class images of indoor and outdoor scenes of everyday objects, such as buildings, tables, cups, and chairs.  The version of ADE20K that we used was grouped into 150 unique ground truth labels.

## Usage
All of our code can be found in the `python/` directory.  Our main hyperparameters for this project consisted of:

* `num_clusters` = number of clusters we use for our K-Means Clustering algorithm
* `num_keypoints` = the maximum number of keypoints we allow for each image for our SIFT algorithm
* `num_topics` = number of topics we use for LDA

For our dataset, our optimal observed tuple of hyperparameters was **(300 clusters, 300 keypoints, and 20 topics)**.

### Running our Pipeline
Running our pipeline can be accomplished using the methods defined in `lda.py`.  This script contains flexible code for running the full pipeline (SIFT, CNN features, K-Means, and LDA), or LDA on pre-clustered features.  

Other Python files, namely `utils.py`, `feature_extraction.py` (extracts SIFT features/can apply K-Means Clustering), `dataset.py` (creates a smooth and flexible Python pipeline), `eval_k_means_call.py` (evaluates our framework using L2 norms on K-Means Clusters and Symmetric KL Divergence on LDA topics), and `validation.py` can also be of use for similar or downstream applications of this unsupervised image clustering framework.

### Baselines
Another major component of this research project was the use of baselines, namely **Principal Component Analysis (PCA)** and **Variational Autoencoders (VAEs)**.  These can be integrated into similar/downstream applications of this framework through `pca.py`, `vae.py`, and `dataset.py`.

**Note:** All information following this is not relevant for the operational use of this repository, and only details the background and motivation for this project.  For more technical/theoretical information on our project, please see our final paper under [final_paper.pdf](https://github.com/rmsander/spatial_LDA/blob/master/final_paper.pdf), and our final poster under [final_poster.pdf](https://github.com/rmsander/spatial_LDA/blob/master/final_poster.pdf).

## Background and Motivation

One of the most consistent roadblocks in many supervised machine learning applications is a lack of labeled data.  Though data in general can be hard to curate, frequently the overarching issue isn't a general lack of data, but a general lack of labeled data. 

This labeled data issue is particularly true for many computer vision tasks - imagery is everywhere, but labeling objects, segmenting scenes, and drawing bounding boxes to train supervised machine learning classifiers is time and resource-intensive. This project seeks to accelerate the machine learning annotation pipeline by using unsupervised machine learning and computer vision algorithms - that is, algorithms that don't rely on the data to already be labeled.  We use these unsupervised machine techniques to find latent, or "hidden" features with which to group our data.

## Finding Latent Features: SIFT, pretrained CNNs, and K-Means CLustering

### Using SIFT to Find Image Keypoints and Gradients
In order to use LDA for clustering images, we need to find features for each of our images that capture the local structure and features of each image.  To implement this feature detection scheme, we use the **Scale-Invariant Feature Transform** (SIFT), a computer vision algorithm designed for feature detection that computes "keypoints" and "local descriptors" of an image.  For more information on SIFT, see the original research paper on it [here](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf).  Each keypoint can be expressed as a 128-dimensional vector, where each element of this vector captures local gradients and variation at that keypoint in the image.  We use our aggregated keypoint matrix as input for our K-Means Clustering algorithm, which we will discuss below.

### Feature Mappings with Pre-Trained Convolutional Neural Networks (CNNs)
Another way in which we discover latent features for intelligent organization of images is through the use of pre-trained Convolutional Neural Networks.  Research has shown that the penultimate and antepenultimate layers of a convolutional neural network (assuming the last layer is a variant of a SoftMax layer) usually best capture the lowest-level features of a dataset, and therefore will be able to best discriminate between latent features.  We leverage the pre-trained convolutional filter activations in these layers as additional inputs to our k-means clustering algorithm, which will be discussed below.

### K-Means Clustering for Discrete Latent Feature Representation
In order to analyze our data with LDA for organizing our images, we need to create categorical labels from our latent features from our continuous latent features.  We accomplish this using K-Means Clustering, which is applied to our SIFT feature matrix and CNN features.  From here, we can apply LDA on our clustered latent features.

### Grouping Images Through Latent Dirichlet Allocation and Discretized Latent Features

Once we have obtained our Visual Bag of Words (VBOW) features through SIFT, CNN filters, and K-Means Clustering, we are now ready to cluster our images using Latent Dirichlet Allocation (LDA).

## Credits
Thank you to the 6.867 team, namely Professor David Sontag and our TA Matthew McDermott, for invaluable feedback and guidance throughout this project.







