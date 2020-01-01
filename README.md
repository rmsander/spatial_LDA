# Spatial Latent Dirichlet Allocation
## Overview
This repository contains the implementation of an unsupervised image organization model for use in semi-automation of the image annotation and data curation process.  It uses Latent Dirichlet Allocation (LDA), Scale-Invariant Feature Transform (SIFT), and pre-trained Convolutional Neural Network (CNN) filters to cluster unlabeled images into different categories using latent features.

## Installation
To install the requirements for this codebase, you can use either `conda` or `pip`.  In the root directory of this codebase (`spatial_LDA/`):
1. `conda env create -f environment.yml`

OR

2. `pip install -r requirements.txt`

## Background and Motivation

One of the most consistent roadblocks in many supervised machine learning applications is a lack of labeled data.  Though data in general can be hard to curate, frequently the overarching issue isn't a general lack of data, but a general lack of labeled data. 

This labeled data issue is particularly true for many computer vision tasks - imagery is everywhere, but labeling objects, segmenting scenes, and drawing bounding boxes to train supervised machine learning classifiers is time and resource-intensive. This project seeks to accelerate the machine learning annotation pipeline by using unsupervised machine learning and computer vision algorithms - that is, algorithms that don't rely on the data to already be labeled.  We use these unsupervised machine techniques to find latent, or "hidden" features with which to group our data.

## Finding Latent Features: SIFT, pretrained CNNs, and K-Means CLustering

### Using SIFT to Find Image Keypoints and Gradients
In order to use LDA for clustering images, we need to find features for each of our images that capture the local structure and features of each image.  To implement this feature detection scheme, we use the **Scale-Invariant Feature Transform** (SIFT), a computer vision algorithm designed for feature detection that computes "keypoints" and "local descriptors" of an image.  For more information on SIFT, see the original research paper on it [here](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf).  Each keypoint can be expressed as a 128-dimensional vector, where each element of this vector captures local gradients and variation at that keypoint in the image.  We use our aggregated keypoint matrix as input for our K-Means Clustering algorithm, which we will discuss below.

### Feature Mappings with Pre-Trained Convolutional Neural Networks (CNNs)
Another way in which we discover latent features for intelligent organization of images is through the use of pre-trained Convolutional Neural Networks.  Research has shown that the penultimate and antepenultimate layers of a convolutional neural network (assuming the last layer is a variant of a SoftMax layer) usually best capture the lowest-level features of a dataset, and therefore will be able to best discriminate between latent features.  We leverage the pre-trained convolutional filter activations in these layers as additional inputs to our k-means clustering algorithm, which will be discussed below.

### K-Means Clustering for Discrete Latent Feature Representation
In order to analyze our data with LDA

## Latent Dirichlet Allocation

The main generative technique we utilize in this codebase is Latent Dirichlet Allocation (LDA), a probabilistic model typically used in natural language processing for grouping documents and words into different topics.  Though its main uses are in NLP, we can leverage it for vision tasks as well, using a Visual Bag of Words (VBOV) representation.
Once we have obtained our Visual Bag of Words (VBOW) features through SIFT, we are now ready to cluster our images using Latent Dirichlet Allocation (LDA).  The procedure for this algorithm is as follows:

<TODO: WRITE UP ALGOTHMIC STEPS FOR LDA>





Where the data lives: /programs/datasets/google_open_image (on Yaateh's box)
