import unittest
import numpy as np
from feature_extraction import get_difference_histograms


class MyTestCase(unittest.TestCase):
    def test_row_vectors(self):
        A = np.array([0.01, 0.99])
        B = np.array([0.99, 0.01])
        KL = get_difference_histograms(A, B, metric="kl")
        print("KL divergence is: {}".format(KL))
        print("Data Type of KL Divergence is: {}".format(type(KL)))

    def test_col_vectors(self):
        A = np.array([0.01, 0.99]).T
        B = np.array([0.99, 0.01]).T
        KL = get_difference_histograms(A, B, metric="kl")
        print("KL divergence is: {}".format(KL))
        print("Data Type of KL Divergence is: {}".format(type(KL)))

if __name__ == '__main__':
    unittest.main()
