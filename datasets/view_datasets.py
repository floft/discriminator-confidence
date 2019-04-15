#!/usr/bin/env python3
"""
As a sanity check, load the data from the source/target domains and display it

Note: probably want to run this prefixed with CUDA_VISIBLE_DEVICES= so that it
doesn't use the GPU (if you're running other jobs).
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


def display(name, images, max_number=16):
    fig = plt.figure(figsize=(4, 4))
    fig.suptitle(name)

    for i, image, in enumerate(images[:max_number]):
        plt.subplot(4, 4, i+1)
        channels = image.shape[2]

        if i == 0:
            print(name, "shape", image.shape, "min", image.min(),
                "max", image.max(), "mean", image.mean())

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
        source_dataset, target_dataset = datasets.load_da(FLAGS.source, FLAGS.target)
    else:
        source_dataset = datasets.load(FLAGS.source)
        target_dataset = None

    source_data = source_dataset.train_images
    target_data = target_dataset.train_images \
        if target_dataset is not None else None

    display("Source", source_data)

    if target_dataset is not None:
        display("Target", target_data)

    plt.show()


if __name__ == "__main__":
    app.run(main)
