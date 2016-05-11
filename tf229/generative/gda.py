"""Implements Gaussian Discriminant Analysis from Stanford 229."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np

def regression(x, y):
    """IImplements Gaussian Discriminant Analysis from Stanford 229.
    (http://cs229.stanford.edu/notes/cs229-notes2.pdf)
    Args:
      x: The covariates or factors of the model in an n by m array (n is number)
        of data points and m is number of factors
      y: The targets or labels of the model in an n by 1 array (either 0 or 1)
    Raises:
      TODO
    Returns:
      A (p, u0, u1, Sigma) tuple, where p is the probability of the target being
      1, u0 is the gaussian mean of the Y=0 distribution, u1 for the Y=1, and
      Sigma is their joint covariance matrix.
    """
    num_predictors = len(x[0])
    num_data = len(x)

    x = np.array(x)
    y = np.array(y)

    print(x)
    x0 = x[np.where( y == 0)]
    print(x0)

    x1 = x[np.where( y == 1)]

    with tf.Graph().as_default() as _:
        Y = tf.placeholder(tf.float32, [num_data])
        X0 = tf.placeholder(tf.float32, [len(x0), num_predictors])
        X1 = tf.placeholder(tf.float32, [len(x1), num_predictors])

        # sum along the num_data axis
        U0 = tf.reduce_mean(X0, 0)
        U1 = tf.reduce_mean(X1, 0)

        Sigma0 = tf.matmul(tf.transpose(X0 - U0), (X0 - U0))
        Sigma1 = tf.matmul(tf.transpose(X1 - U1), (X1 - U1))
        Sigma = (Sigma1 + Sigma0)/num_data

        P = tf.reduce_mean(Y)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            p, u0, u1, sigma = sess.run(
                [P, U0, U1, Sigma],
                feed_dict={X0:x0, X1: x1, Y:y})

            Parameters = namedtuple("Parameters", ["Probability_of_target",
                "Gaussian_mean_of_the_Y0_distribution",
                "Gaussian_mean_of_the_Y1_distribution",
                "Joint_covariance_matrix"])
            return Parameters(p, u0, u1, sigma)


if __name__ == "__main__":
    X_TEST = np.reshape([[0.5, 0.2], [0.5, 0.2], [0.5, 0.2],
        [0.25, 1.0], [0.25, 1.0], [0.25, 1.0]], (6, 2))
    Y_TEST = np.reshape([0, 0, 0, 1, 1, 1], (6))


    print(regression(X_TEST, Y_TEST))
