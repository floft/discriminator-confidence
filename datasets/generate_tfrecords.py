#!/usr/bin/env python3
"""
Create all the tfrecord files

For some source domain A and target domain B, where C and D are A and B but in
alphabetical order:
    C_and_D_A_{train,valid,test}.tfrecord
    C_and_D_B_{train,valid,test}.tfrecord

For example for MNIST to MNIST-M or MNIST-M to MNIST (since both ways use the
same data):
    mnist_and_mnistm_mnist_{train,valid,test}.tfrecord
    mnist_and_mnistm_mnistm_{train,valid,test}.tfrecord

We do this because otherwise for some domains like SynNumbers to SVHN we use
nearly all of my 32 GiB of RAM just loading the datasets and it takes a while
as well.
"""
import os
import numpy as np
import tensorflow as tf

import datasets

from tfrecord import write_tfrecord, tfrecord_filename


def write(filename, x, y):
    if not os.path.exists(filename):
        write_tfrecord(filename, x, y)
    else:
        print("Skipping:", filename, "(already exists)")


def shuffle_together_calc(length, seed=None):
    """ Generate indices of numpy array shuffling, then do x[p] """
    rand = np.random.RandomState(seed)
    p = rand.permutation(length)
    return p


def to_numpy(value):
    """ Make sure value is numpy array """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return value


def valid_split(images, labels, seed=None, validation_size=1000):
    """ Split test data into validation/test as is commonly done, taking 1000
    random samples for a validation set """
    assert len(images) == len(labels), "len(images) != len(labels)"
    p = shuffle_together_calc(len(images), seed=seed)
    images = to_numpy(images)[p]
    labels = to_numpy(labels)[p]

    valid_images = images[:validation_size]
    valid_labels = labels[:validation_size]
    test_images = images[validation_size:]
    test_labels = labels[validation_size:]

    return valid_images, valid_labels, test_images, test_labels


def save_adaptation(source, target, seed=0):
    """ Save single source-target pair datasets """
    source_dataset, target_dataset = datasets.load_da(source, target)

    source_valid_images, source_valid_labels, \
        source_test_images, source_test_labels = \
        valid_split(source_dataset.test_images, source_dataset.test_labels,
            seed=0)

    write(tfrecord_filename(source, target, source, "train"),
        source_dataset.train_images, source_dataset.train_labels)
    write(tfrecord_filename(source, target, source, "valid"),
        source_valid_images, source_valid_labels)
    write(tfrecord_filename(source, target, source, "test"),
        source_test_images, source_test_labels)

    target_valid_images, target_valid_labels, \
        target_test_images, target_test_labels = \
        valid_split(target_dataset.test_images, target_dataset.test_labels,
            seed=1)

    write(tfrecord_filename(source, target, target, "train"),
        target_dataset.train_images, target_dataset.train_labels)
    write(tfrecord_filename(source, target, target, "valid"),
        target_valid_images, target_valid_labels)
    write(tfrecord_filename(source, target, target, "test"),
        target_test_images, target_test_labels)


if __name__ == "__main__":
    # Only list one direction since the other direction uses the same data
    adaptation_problems = [
        ("mnist", "usps"),
        ("svhn", "mnist"),
        ("mnist", "mnistm"),
        ("synnumbers", "svhn"),
    ]

    # Save tfrecord files for each of the adaptation problems
    for source, target in adaptation_problems:
        save_adaptation(source, target)
