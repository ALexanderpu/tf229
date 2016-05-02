"""Implements the naive bayes algorithm from Stanford 229."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np

def regression(x, y):
    """Implements the naive bayes algorithm from Stanford 229
     (http://cs229.stanford.edu/notes/cs229-notes2.pdf)
    Args:
      x: The covariates or factors of the model in an n by m array (n is number)
        of binary data points and m is number of factors
      y: The targets or labels of the model in an n by c array, where c is the
        number of classes. y is a list of one hot vectors.
    Raises:
      TODO
    Returns:
      A (Phis, Ps) tuple, where the Phis are a predictors by classes matrix for
        the prob of predictor given class. And the Ps are the prob of a class
    """
    num_data = len(x)
    num_predictors = len(x[0])
    num_classes = len(y[0])
    x = np.array(x)
    y = np.array(y)

    with tf.Graph().as_default() as _:
        X = tf.placeholder(tf.float32, [num_data, num_predictors])
        Y = tf.placeholder(tf.float32, [num_data, num_classes])

        # num_predictors,num_data * num_data,num_classes => num_predictors,num_classes
        # we add one to each of the counts ~ Laplace Smoothing
        summed_occurances = tf.matmul(tf.transpose(X), Y) + 1
        # add the number of classes to each ~ Laplace Smoothing
        normalization = tf.reduce_sum(Y, 0) + num_classes

        # sum along the num_predictors axis => batch_size,1
        Phis = summed_occurances / normalization

        Ps = tf.reduce_mean(Y, 0)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            phis, ps = sess.run(
                [Phis, Ps],
                feed_dict={X:x, Y:y})

            Parameters = namedtuple("Parameters", ["Phis", "Ps"])
            return Parameters(phis, ps)


if __name__ == "__main__":
    X_TEST = np.array([[0],[1],[1],[1],[1]])
    # init one hots
    Y_TEST = np.zeros((5,3))
    Y_TEST[np.arange(5), [0,1,2,1,2]] = 1

    print(regression(X_TEST, Y_TEST))
