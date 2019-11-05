"""This file contains code for a latent dirichlet allocation (SLDA),
which uses principles from maximum likelihood estimation and
Expectation-Maximization."""

# Native Python imports
import os
import copy

# External package imports
import numpy as np
import cv2 as cv

# External package imports
import numpy as np
import cv2 as cv
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Custom module imports
import dataset
import feature_extraction


class LDA:
    """Class that implements Latent Dirichlet Allocation using an
    Expectation-Maximization framework.  This function iterates through the
    following two steps to find a locally-optimal maximum likelihood estimate of
    the relevant posterior distribution.

    Arguments:
        features: A numpy array of size (N_images x N_descriptors)
        alpha (float): A parameter for our LDA model (TODO: add on here).
        beta (float): A parameter for our LDA model (TODO: add on here).

    """
    def __init__(self, data_path, alpha=1, beta=1, eps=1e-5,n_topics=10):
        self.data_path = data_path  # File path for data
        self.alpha = alpha  # Dirichlet dist hyperparameter
        self.beta = beta  # Dirichlet dist hyperparameter
        self.log_likelihood = None  # Array of log likelihoods
        self.parameters = None  # Numpy vector of parameters
        self.eps = eps  # Convergence threshold
        self.keypoints = None
        self.n_topics = n_topics

    def get_data_matrix(self):
        self.M = np.load(self.feature_path)

    def off_the_shelf_LDA(self):
        lda = LDA(n_components=self.n_topics)
        lda.fit(self.M)

    def sample_phi_from_dirichlet(self, n=1):
        """Function to sample from a Dirichlet distribution.

        Arguments:
            n (int): The number of samples from the Dirichlet distribution
                this function returns.

        Returns:
            A np array of samples from the Dirichlet distribution, of size n.
        """

        return np.random.dirichlet(self.beta, size=n)

    def sample_pi_from_dirichlet(self, n=1):
        """Function to sample from a Dirichlet distribution.

        Arguments:
            n (int): The number of samples from the Dirichlet distribution
                this function returns.

        Returns:
            A np array of samples from the Dirichlet distribution, of size n.
        """

        return np.random.dirichlet(self.alpha, size=n)

    def expectation_step(self):
        """This first step of the Expectation-Maximization algorithm computes
        the expectation over the distributions of interest, holding several 
        parameters fixed""".

        pass

    def maximization_step(self):
        """This second step of the Expectation-Maximization algorithm finds
        a set of parameters that optimizes the log likelihood of the
        distributions of interest (i.e. Maximum Likelihood Estimation)."""

        self.parameters = np.argmax(self.log_likelihood)  # Update parameters

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


def main():
    #TODO: FILL IN feature_path
    feature_path = ""
    lda = LDA(feature_path)  # Make the class
    lda.get_data_matrix()    # Import the features
    lda.off_the_shelf_LDA()  # Fit the sklearn LDA model

    # Now we can predict!
