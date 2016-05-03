"""Implements Gaussian Discriminant Analysis from Stanford 229."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np

def train(x, y):
    """IImplements Gaussian Discriminant Analysis from Stanford 229. This could
    further be extened to cover k classes
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

    x0 = x[np.where( y == 0)]
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

def predict(x, p, u0, u1, sigma):
    """"Predicts targets using a batch of predictors and a set of parameters
    as inputs
    Args:
      x: The covariates or factors of the model in an m array where m is number
      of factors
      p: the probablity of class 1
      u0: the mean of class 0
      u1: the mean of class 1
      sigma: the shared cov matrix
    Raises:
      TODO
    Returns:
      a 0 or a 1 depending on which class matches
    """
    num_predictors = len(x)
    x = np.reshape(x, [1, num_predictors])
    u0 = np.reshape(u0, [1, num_predictors])
    u1 = np.reshape(u1, [1, num_predictors])

    with tf.Graph().as_default() as _:
        X = tf.placeholder(tf.float32, [1, num_predictors])

        U0 = tf.placeholder(tf.float32, [1, num_predictors])
        U1 = tf.placeholder(tf.float32, [1, num_predictors])

        Sigma = tf.placeholder(tf.float32, [num_predictors, num_predictors])

        P = tf.placeholder(tf.float32, [1])

        log_P0 = tf.matmul( tf.matmul(X, tf.matrix_inverse(Sigma)),
            tf.transpose(U0)) - 0.5 * tf.matmul(tf.matmul(U0,tf.matrix_inverse(Sigma)),
            tf.transpose(U0)) + tf.log(1 - P)
        log_P1 = tf.matmul(tf.matmul(X,tf.matrix_inverse(Sigma)),
            tf.transpose(U1)) - 0.5 * tf.matmul(tf.matmul(U1,tf.matrix_inverse(Sigma)),
            tf.transpose(U1)) + tf.log(P)

        P0 = tf.exp(log_P0)/(tf.exp(log_P0) + tf.exp(log_P1))
        P1 = tf.exp(log_P1)/(tf.exp(log_P0) + tf.exp(log_P1))

        with tf.Session() as sess:

            p0, p1 = sess.run([log_P0, log_P1],
                feed_dict={X:x, P:p, U0:u0, U1:u1, Sigma:sigma})

            return int(p0 < p1)

if __name__ == "__main__":
    X_TEST = np.reshape([[0.51, 0.2], [0.55, 0.21], [0.52, 0.22],
        [0.24, 1.01], [0.255, 1.07], [0.259, 1.03]], (6, 2))
    Y_TEST = np.reshape([0, 0, 0, 1, 1, 1], (6))


    P_, U0_, U1_, SIGMA = train(X_TEST, Y_TEST)

    print(P_, U0_, U1_, SIGMA)

    print(predict(X_TEST[0], [P_], U0_, U1_, SIGMA))
