"""Implements stochastic gradient decent on a perceptron as seen in
Stanford 229."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np

def regression(x, y, **kwargs):
    """Implements stochastic gradient decent for a perceptron as seen in
    Stanford 229 (http://cs229.stanford.edu/notes/cs229-notes1.pdf)
    Args:
      x: The covariates or factors of the model in an n by m array (n is number)
        of data points and m is number of factors
      y: The targets or labels of the model in an n by 1 array
      kwargs:
        iterations: The number of steps to train
        batch_size: The number of samples to use per step
        verbosity_step: The number of steps between each printout of the cost
            of the model (negative for no printouts)
        step_size: The distance we step down the gradient each step
        seed: the seed for choosing our batches (0 if no seed)
    Raises:
      TODO
    Returns:
      A (Weights, Bias) tuple
    """
    # extract the kwargs
    iterations = kwargs.get("iterations", 100)
    batch_size = kwargs.get("batch_size", 10)
    verbosity_step = kwargs.get("verbosity_step", 20)
    step_size = kwargs.get("step_size", 10e-1)
    seed = kwargs.get("seed", 0)

    if seed:
        np.random.seed(seed)

    num_predictors = len(x[0])
    x = np.array(x)
    y = np.array(y)

    with tf.Graph().as_default() as _:
        X = tf.placeholder(tf.float32, [batch_size, num_predictors])
        Y = tf.placeholder(tf.float32, [batch_size, 1])

        W = tf.Variable(tf.zeros([num_predictors, 1]))
        b = tf.Variable(1.0)

        # if > 0 -> 1 else 0
        thresholded = (tf.nn.softsign(tf.matmul(X, W) + b) + 1) / 2
        cost = tf.reduce_sum(tf.square(Y - thresholded))

        train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            for i in range(iterations):
                sample_indexes = np.random.choice(len(y), batch_size)
                sample_xs = x[sample_indexes]
                sample_ys = y[sample_indexes]

                weights, bais, batch_cost, _ = sess.run(
                    [W, b, cost, train_step],
                    feed_dict={X:sample_xs, Y:sample_ys})

                if i % verbosity_step == 0:
                    print(batch_cost)

            Parameters = namedtuple("Parameters", ["Weights", "Biases"])
            return Parameters(weights, bais)


if __name__ == "__main__":
    X_TEST = np.reshape([0.5, 0.5, 0.5, 0.25, 0.25, 0.25], (6, 1))
    Y_TEST = np.reshape([0, 0, 0, 1, 1, 1], (6, 1))

    print(regression(X_TEST, Y_TEST, iterations=3000))
