# Spatial Latent Dirichlet Allocation

This repository contains the implementation of an image-based LDA model for use in semi-automation of the image annotation and data curation process.  It uses unsupervised Latent Dirichlet Allocation (LDA) and Scale-Invariant Feature Transform (SIFT) algorithms to cluster unlabeled images into different categories.  

# Installation
To install the requirements for this codebase, you can use either `conda` or `pip`.  In the root directory of this codebase (`spatial_LDA/`):
1. `conda env create -f environment.yml`

Or

2. `pip install -r requirements.txt`

# Background and Motivation

One of the most consistent roadblocks in many supervised machine learning applications is a lack of labeled data.  Though data in general can be hard to curate, frequently the overarching issue isn't a general lack of data, but a general lack of labeled data. 

This labeled data issue is particularly true for many computer vision tasks - imagery is everywhere, but labeling objects, segmenting scenes, and drawing bounding boxes to train supervised machine learning classifiers is time and resource-intensive. This project seeks to accelerate the machine learning annotation pipeline by using unsupervised machine learning and computer algorithms - that is, algorithms that don't rely on the data to already be labeled.  

# Overview

The main technique we utilize in this codebase is Latent Dirichlet Allocation (LDA), a probabilistic model typically used in natural language processing for grouping documents and words into different topics.  Though its main uses are in NLP, we can leverage it for vision tasks as well, using a Visual Bag of Words (VBOV) representation.

## Visual Bag of Words (VBOW)

In order to use LDA for clustering images, we need to find features for each of our images that capture the local structure and features of each image.  To implement this feature detection scheme, we use the **Scale-Invariant Feature Transform** (SIFT), a computer vision algorithm designed for feature detection that computes "keypoints" and "local descriptors" of an image.  This algorithm is as follows:

<TODO: WRITE UP ALGORITHMIC STEPS FOR SIFT>

## Latent Dirichlet Allocation

Once we have obtained our Visual Bag of Words (VBOW) features through SIFT, we are now ready to cluster our images using Latent Dirichlet Allocation (LDA).  The procedure for this algorithm is as follows:

<TODO: WRITE UP ALGOTHMIC STEPS FOR LDA>





Where the data lives: /programs/datasets/google_open_image (on Yaateh's box)
