"""Implements the naive bayes algorithm from Stanford 229."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np

def train(x, y, model_path=""):
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

        Phis = summed_occurances / normalization

        Ps = tf.reduce_mean(Y, 0)

        saver = tf.train.Saver([Phis, Ps])

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            phis, ps = sess.run(
                [Phis, Ps],
                feed_dict={X:x, Y:y})

            if model_path:
                saver.save(sess, model_path)

            Parameters = namedtuple("Parameters", ["Phis", "Ps"])
            return Parameters(phis, ps)


def predict(x, num_classes, model_path):
    num_predictors = len(x)

    with tf.Graph().as_default() as _:
        X = tf.placeholder(tf.float32, [num_predictors])

        Phis = tf.Variable([num_predictors, num_classes])

        Ps = tf.Variable([num_classes])

        saver = tf.train.Saver([Phis, Ps])


if __name__ == "__main__":
    X_TEST = np.array([[0],[1],[1],[1],[1]])
    # init one hots
    Y_TEST = np.zeros((5,3))
    Y_TEST[np.arange(5), [0,1,2,1,2]] = 1

    print(train(X_TEST, Y_TEST))

import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.constant([1.0,0.0,1.0])

y = tf.ones((5,3))

z = x * y

z.eval()
