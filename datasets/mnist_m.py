#!/usr/bin/env python3
"""
Create a tfrecord file for MNIST-M
"""
import os
import tarfile
import tensorflow as tf

from tfrecord import write_tfrecord


def load_data(compressed="mnist_m.tar.gz"):
    # Check that the tar.gz file exists
    if not os.path.exists(compressed):
        print("Download the 'unpacked version of MNIST-M' mnist_m.tar.gz "
            "from http://yaroslav.ganin.net/")

    # Indexed by image filename, e.g. 00022776.png
    test_labels = None
    train_labels = None
    train_images = {}
    test_images = {}

    # Get data from the file
    tar = tarfile.open(compressed, "r:gz")

    for member in tar.getmembers():
        f = tar.extractfile(member)
        if f is not None:
            folder, filename = os.path.split(member.name.replace("mnist_m/", ""))
            content = f.read()

            if folder == "mnist_m_train":
                train_images[filename] = get_image(content)
            elif folder == "mnist_m_test":
                test_images[filename] = get_image(content)
            elif folder == "" and filename == "mnist_m_train_labels.txt":
                train_labels = get_labels(content)
            elif folder == "" and filename == "mnist_m_test_labels.txt":
                test_labels = get_labels(content)

    assert test_labels is not None and train_labels is not None, \
        "Could not find mnist_m_{train,test}_labels.txt in "+compressed+" file" \
        + "Are you sure you downloaded the unpacked version?"

    assert len(train_images) == len(train_labels), \
        "train_images and train_labels are of different sizes"
    assert len(test_images) == len(test_labels), \
        "test_images and test_labels are of different sizes"

    # Create x and y lists
    train_x, train_y = get_xy(train_images, train_labels)
    test_x, test_y = get_xy(test_images, test_labels)

    return train_x, train_y, test_x, test_y


def get_labels(content):
    """ Read data in the format "image_name.png label_number", one per line,
    into a dictionary {"image_name.png": label_number, ...} """
    labels = {}
    lines = content.decode("utf-8").strip().split("\n")

    for line in lines:
        name, label = line.split(" ")

        try:
            label = int(label)
        except ValueError:
            raise ValueError("could not parse label as integer")

        labels[name] = label

    return labels


def get_image(content):
    """ Use TensorFlow to decode the PNG images into a tensor """
    return tf.io.decode_image(content)


def get_xy(images, labels):
    """ Take the image and labels dictionaries and create x and y lists where
    elements correspond """
    x = []
    y = []
    keys = list(images.keys())
    keys.sort()

    for k in keys:
        x.append(images[k])
        y.append(labels[k])

    return x, y


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = load_data()
    write_tfrecord("mnist_m_train.tfrecord", train_x, train_y)
    write_tfrecord("mnist_m_test.tfrecord", test_x, test_y)
