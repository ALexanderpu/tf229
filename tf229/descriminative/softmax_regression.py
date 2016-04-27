"""Implements stochastic gradient decent on logistic regression as seen in
Stanford 229."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np

def regression(x, y, **kwargs):
    """Implements stochastic gradient decent on softmax regression as seen in
    Stanford 229 (http://cs229.stanford.edu/notes/cs229-notes1.pdf)
    Args:
      x: The covariates or factors of the model in an n by m array (n is number)
        of data points and m is number of factors
      y: The targets or labels of the model in an n by c array, where c is the
        number of classes
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
    num_classes = len(y[0])
    x = np.array(x)
    y = np.array(y)

    with tf.Graph().as_default() as _:
        X = tf.placeholder(tf.float32, [batch_size, num_predictors])
        Y = tf.placeholder(tf.float32, [batch_size, num_classes])

        Ws = tf.Variable(tf.truncated_normal([num_predictors,num_classes], stddev=0.001))
        bs = tf.Variable(tf.truncated_normal([1,num_classes], stddev=0.001))

        # batch_size,num_predictors * num_predictors,num_classes => batch_size,num_classes
        weighted_X = tf.matmul(X, Ws) + bs

        # sum along the num_predictors axis => batch_size,1
        normalization = tf.reduce_sum(tf.exp(weighted_X), 1, keep_dims=True)

        # subtracts the batch normalization from each term in the batch
        logits = weighted_X - tf.log(normalization)

        # elementwise multiply. logits and Y are the same shape
        # this will only keep the logits that are true
        cost = -tf.reduce_mean(Y * logits)

        train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cost)

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            for i in xrange(iterations):
                sample_indexes = np.random.choice(len(y), batch_size)
                sample_xs = x[sample_indexes]
                sample_ys = y[sample_indexes]

                weights, biases, step_cost, _ = sess.run(
                    [Ws, bs, cost, train_step],
                    feed_dict={X:sample_xs, Y:sample_ys})

                if i % verbosity_step == 0:
                    print(step_cost)


            Parameters = namedtuple("Parameters", ["Weights", "Biases"])
            return Parameters(weights, biases)


if __name__ == "__main__":
    X_TEST = np.array([[0],[1],[2],[1],[2]])
    # init one hots
    Y_TEST = np.zeros((5,3))
    Y_TEST[np.arange(5), [0,1,2,1,2]] = 1

    print(regression(X_TEST, Y_TEST, iterations=2000, batch_size=5))
