"""
Domain adaptation utils
"""
import numpy as np
import tensorflow as tf


def tf_domain_labels(label, batch_size, num_domains=2):
    """ Generate one-hot encoded labels for which domain data is from (using TensorFlow) """
    return tf.tile(tf.one_hot([label], depth=num_domains), [batch_size, 1])


def domain_labels(label, batch_size, num_domains=2):
    """ Generate one-hot encoded labels for which domain data is from (using numpy) """
    return np.tile(np.eye(num_domains)[label], [batch_size, 1])
