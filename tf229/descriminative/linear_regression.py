"""Implements stochastic gradient decent linear regression as seen in
Stanford 229."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple

import tensorflow as tf
import numpy as np

def train(x, y, **kwargs):
    """Implements stochastic gradient decent linear regression as seen in
    Stanford 229
    Args:
      x: The covariates or factors of the model in an n by m array (n is number)
        of data points and m is number of factors
      y: The targets or labels of the model in an n by 1 array
      kwargs:
        model_path: the location where the tf model file should be saved
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
    model_path = kwargs.get("model_path", "")
    iterations = kwargs.get("iterations", 100)
    batch_size = kwargs.get("batch_size", 10)
    verbosity_step = kwargs.get("verbosity_step", 20)
    step_size = kwargs.get("step_size", 10e-6)
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

        saver = tf.train.Saver([W, b])

        cost = tf.reduce_sum(tf.square((tf.matmul(X, W) + b) - Y))
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

            if model_path:
                saver.save(sess, model_path)

            Parameters = namedtuple("Parameters", ["Weights", "Biases"])
            return Parameters(weights, bais)

def predict(x, model_path):
    """Predicts targets using a batch of predictors and a model trained by
    the linear regress train method
    Args:
      x: The covariates or factors of the model in an n by m array (n is number)
        of data points and m is number of factors
      model_path: location of the tf model file
    Raises:
      TODO
    Returns:
      a num data by 1 array of predictions
    """
    num_data = len(x)
    num_predictors = len(x[0])

    x = np.array(x)

    with tf.Graph().as_default() as _:
        X = tf.placeholder(tf.float32, [num_data, num_predictors])

        W = tf.Variable(tf.zeros([num_predictors, 1]))
        b = tf.Variable(1.0)

        saver = tf.train.Saver([W, b])

        Predictions = tf.matmul(X, W) + b

        with tf.Session() as sess:
            saver.restore(sess, model_path)

            predictions = sess.run([Predictions], feed_dict={X:x})

            return predictions


if __name__ == "__main__":
    X_TEST = np.reshape(range(1, 11), (10, 1))
    Y_TEST = np.reshape(range(3, 22, 2), (10, 1))

    print(train(X_TEST, Y_TEST, iterations=1000,
        model_path="models/linear_regression/linear_regression"))

    print(predict(X_TEST, "models/linear_regression/linear_regression"))
