#!/usr/bin/env python3
"""
As a sanity check, load the data from the source/target domains and display it
"""
import tensorflow as tf
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from tensorflow.python.framework import config as tfconfig

import datasets

FLAGS = flags.FLAGS

flags.DEFINE_enum("source", None, datasets.names(), "What dataset to use as the source")
flags.DEFINE_enum("target", "", [""]+datasets.names(), "What dataset to use as the target")
flags.DEFINE_float("gpumem", 0.1, "Percentage of GPU memory to let TensorFlow use")

flags.mark_flag_as_required("source")


def display(name, images):
    fig = plt.figure(figsize=(4,4))
    fig.suptitle(name)

    for i, image, in enumerate(images):
        plt.subplot(4, 4, i+1)
        channels = image.shape[2]

        if channels == 1:
            plt.imshow(image[:, :, 0] * 127.5 + 127.5, cmap='gray')
        elif channels == 3:
            plt.imshow(tf.cast(image[:, :, :] * 127.5 + 127.5, tf.int32))
        else:
            raise NotImplementedError("display() only supports gray or RGB")

        plt.axis('off')


def main(argv):
    # Allow running multiple at once
    # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
    # https://github.com/tensorflow/tensorflow/issues/25138
    # Note: GPU options must be set at program startup
    tfconfig.set_gpu_per_process_memory_fraction(FLAGS.gpumem)

    # Input data
    if FLAGS.target != "":
        source_dataset, target_dataset = datasets.load_da(FLAGS.source,
            FLAGS.target, train_batch=16)
    else:
        source_dataset = datasets.load(FLAGS.source, train_batch=16)
        target_dataset = None

    source_iter = iter(source_dataset.train)
    target_iter = iter(target_dataset.train) \
        if target_dataset is not None else None
    source_batch = next(source_iter)
    target_batch = next(target_iter) \
        if target_dataset is not None else None

    # Display a batch of data ([0] is the image, [1] is the label)
    display("Source", source_batch[0])

    if target_dataset is not None:
        display("Target", target_batch[0])

    plt.show()


if __name__ == "__main__":
    app.run(main)
