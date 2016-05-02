"""Implements the naive bayes algorithm from Stanford 229."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np

def train(x, y):
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
      A (Phis, Pys, Pxs) tuple, where the Phis are a predictors by classes matrix for
        the prob of predictor given class. And the Pys are the prob of a class
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

        Pys = tf.reduce_mean(Y, 0)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            phis, pys = sess.run(
                [Phis, Pys],
                feed_dict={X:x, Y:y})


            Parameters = namedtuple("Parameters", ["Phis", "Pys"])
            return Parameters(phis, pys)


def predict(x, phis, pys):
    """"Predicts targets using a batch of predictors and a set of parameters
    as inputs
    Args:
      x: The covariates or factors of the model in an m array where m is number
      of factors
      phis: the phis parameter from a trained model
      pys: the pys parameter from a trained model
    Raises:
      TODO
    Returns:
      a num data by 1 array of predictions
    """
    num_predictors = len(x)
    num_classes = len(phis[0])

    with tf.Graph().as_default() as _:
        X = tf.placeholder(tf.float32, [num_predictors])

        Phis = tf.placeholder(tf.float32, [num_predictors, num_classes])

        Pys = tf.placeholder(tf.float32, [num_classes])

        # log the phis (they will not be 0 because of laplace smoothing)
        # multiply by the single datapoint, this will zero out all the probs
        # where x is not represented
        conditional = tf.reduce_sum(X * tf.log(tf.transpose(Phis)), 1)

        # add the logged P(y) and the subtract off the total P(x)
        unnormed_prediction = tf.exp(conditional + tf.log(Pys))

        # exponentiate to get the predictions back out of the calculation
        Predictions = unnormed_prediction / tf.reduce_sum(unnormed_prediction)

        with tf.Session() as sess:

            predictions = sess.run([Predictions],
                feed_dict={X:x, Phis:phis, Pys:pys})

            return predictions


if __name__ == "__main__":
    X_TEST = np.array([[0,1,0],[1,0,0],[0,1,1],[1,0,0],[0,1,1]])
    # init one hots
    Y_TEST = np.zeros((5,3))
    Y_TEST[np.arange(5), [0,1,2,1,2]] = 1

    PHIS, PYS = train(X_TEST, Y_TEST)

    print(PHIS, PYS)

    print(predict(X_TEST[2], PHIS, PYS))
