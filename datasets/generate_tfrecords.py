#!/usr/bin/env python3
"""
Create all the tfrecord files

For some source domain A and target domain B, where C and D are A and B but in
alphabetical order:
    C_and_D_A_{train,test}.tfrecord
    C_and_D_B_{train,test}.tfrecord

For example for MNIST to MNIST-M or MNIST-M to MNIST (since both ways use the
same data):
    mnist_and_mnistm_mnist_{train,test}.tfrecord
    mnist_and_mnistm_mnistm_{train,test}.tfrecord

We do this because otherwise for some domains like SynNumbers to SVHN we use
nearly all of my 32 GiB of RAM just loading the datasets and it takes a while
as well.
"""
import os

import datasets

from tfrecord import write_tfrecord, tfrecord_filename


def write(filename, x, y):
    if not os.path.exists(filename):
        write_tfrecord(filename, x, y)
    else:
        print("Skipping:", filename, "(already exists)")


def save_adaptation(source, target):
    source_dataset, target_dataset = datasets.load_da(source, target)

    write(tfrecord_filename(source, target, source, "train"),
        source_dataset.train_images, source_dataset.train_labels)
    write(tfrecord_filename(source, target, source, "test"),
        source_dataset.test_images, source_dataset.test_labels)

    write(tfrecord_filename(source, target, target, "train"),
        target_dataset.train_images, target_dataset.train_labels)
    write(tfrecord_filename(source, target, target, "test"),
        target_dataset.test_images, target_dataset.test_labels)


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
