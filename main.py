#!/usr/bin/env python3
"""
Adversarial Confidence for Domain Adaptation Pseudo-Labeling
"""
import os
import time
import tensorflow as tf

from absl import app
from absl import flags
from absl import logging
from tensorflow.python.framework import config as tfconfig

import models
import load_datasets

from metrics import Metrics
from checkpoints import CheckpointManager
from utils import domain_labels
from file_utils import last_modified_number, write_finished

FLAGS = flags.FLAGS

flags.DEFINE_enum("model", None, models.names(), "What model type to use")
flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
flags.DEFINE_boolean("adapt", False, "Perform domain adaptation on the model")
flags.DEFINE_enum("source", None, load_datasets.names(), "What dataset to use as the source")
flags.DEFINE_enum("target", "", [""]+load_datasets.names(), "What dataset to use as the target")
flags.DEFINE_integer("steps", 100000, "Number of training steps to run")
flags.DEFINE_float("lr", 0.001, "Learning rate for training")
flags.DEFINE_float("lr_mult", 1.0, "Multiplier for extra discriminator training learning rate")
flags.DEFINE_float("gpumem", 0.3, "Percentage of GPU memory to let TensorFlow use")
flags.DEFINE_integer("model_steps", 4000, "Save the model every so many steps")
flags.DEFINE_integer("log_train_steps", 500, "Log training information every so many steps")
flags.DEFINE_integer("log_val_steps", 4000, "Log validation information every so many steps (also saves model)")
flags.DEFINE_boolean("debug", False, "Start new log/model/images rather than continuing from previous run")
flags.DEFINE_integer("debugnum", -1, "Specify exact log/model/images number to use rather than incrementing from last. (Don't pass both this and --debug at the same time.)")

flags.mark_flag_as_required("model")
flags.mark_flag_as_required("source")


def get_directory_names():
    """ Figure out the log and model directory names """
    prefix = FLAGS.source+"-"+FLAGS.target+"-"+FLAGS.model

    if FLAGS.adapt:
        prefix += "-da"

    # Use the number specified on the command line (higher precedence than --debug)
    if FLAGS.debugnum >= 0:
        attempt = FLAGS.debugnum
        logging.info("Debugging attempt: %s", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    # Find last one, increment number
    elif FLAGS.debug:
        attempt = last_modified_number(FLAGS.logdir, prefix+"*")
        attempt = attempt+1 if attempt is not None else 1
        logging.info("Debugging attempt: %s", attempt)

        prefix += "-"+str(attempt)
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)
    # If no debugging modes, use the model and log directory with only the "prefix"
    # (even though it's not actually a prefix in this case, it's the whole name)
    else:
        model_dir = os.path.join(FLAGS.modeldir, prefix)
        log_dir = os.path.join(FLAGS.logdir, prefix)

    return model_dir, log_dir


@tf.function
def train_step(data_a, data_b, model, opt, d_opt, source_domain, target_domain,
        task_loss, domain_loss):
    """ Compiled training step that we call many times """
    if data_a is not None:
        x_a, y_a = data_a
    if data_b is not None:
        x_b, y_b = data_b

    # Concatenate for adaptation - concatenate source labels with all-zero
    # labels for target since we can't use the target labels during
    # unsupervised domain adaptation
    if FLAGS.adapt:
        assert data_a is not None and data_b is not None, \
            "Adaptation requires both datasets A and B"
        x = tf.concat((x_a, x_b), axis=0)
        task_y_true = tf.concat((y_a, tf.zeros_like(y_b)), axis=0)
        domain_y_true = tf.concat((source_domain, target_domain), axis=0)
    else:
        x = x_a
        task_y_true = y_a
        domain_y_true = source_domain

    # Run data through model and compute loss
    with tf.GradientTape() as tape, tf.GradientTape() as d_tape:
        task_y_pred, domain_y_pred = model(x, training=True)

        d_loss = domain_loss(domain_y_true, domain_y_pred)
        loss = task_loss(task_y_true, task_y_pred, training=True) + d_loss

    # Only update domain classifier during adaptation
    if FLAGS.adapt:
        trainable_vars = model.trainable_variables
    else:
        trainable_vars = model.trainable_variables_exclude_domain

    # Update model
    grad = tape.gradient(loss, trainable_vars)
    opt.apply_gradients(zip(grad, trainable_vars))

    # Update discriminator
    if FLAGS.adapt:
        d_grad = d_tape.gradient(d_loss, model.domain_classifier.trainable_variables)
        d_opt.apply_gradients(zip(d_grad, model.domain_classifier.trainable_variables))
    # if FLAGS.adapt:  # TODO try
    #     for _ in range(FLAGS.max_domain_iters):
    #         with tf.GradientTape() as d_tape:
    #             task_y_pred, domain_y_pred = model(x, grl_lambda=0.0, training=True)
    #             d_loss = domain_loss(domain_y_true, domain_y_pred)

    #         d_grad = d_tape.gradient(d_loss, model.domain_classifier.trainable_variables)
    #         d_opt.apply_gradients(zip(d_grad, model.domain_classifier.trainable_variables))

    #         # Break if high enough accuracy
    #         domain_acc = compute_accuracy(domain_y_true, domain_y_pred)
    #         if domain_acc > FLAGS.min_domain_accuracy:
    #             break


def main(argv):
    # Allow running multiple at once
    # https://www.tensorflow.org/guide/using_gpu#allowing_gpu_memory_growth
    # https://github.com/tensorflow/tensorflow/issues/25138
    # Note: GPU options must be set at program startup
    tfconfig.set_gpu_per_process_memory_fraction(FLAGS.gpumem)

    # Figure out the log and model directory filenames
    model_dir, log_dir = get_directory_names()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # For adaptation, we'll be concatenating together half source and half target
    # data, so to keep the batch_size about the same, we'll cut it in half
    train_batch = FLAGS.train_batch

    if FLAGS.adapt:
        train_batch = train_batch // 2

    # Input training data
    #
    # Note: "It is worth noting that only the training sets of the small image
    # datasets were used during training; the test sets usedfor reporting scores
    # only." (self-ensembling) -- so, only use *_test for evaluation.
    if FLAGS.target != "":
        source_dataset, target_dataset = load_datasets.load_da(FLAGS.source,
            FLAGS.target, train_batch=train_batch)
        assert source_dataset.num_classes == target_dataset.num_classes, \
            "Adapting from source to target with different classes not supported"
    else:
        raise NotImplementedError("currently don't support only source")
        source_dataset = load_datasets.load(FLAGS.source, train_batch=train_batch)
        target_dataset = None

    # Iterator and evaluation datasets if we have the dataset
    source_iter = iter(source_dataset.train)
    source_dataset_eval = source_dataset.test_evaluation
    target_iter = iter(target_dataset.train) \
        if target_dataset is not None else None
    target_dataset_eval = target_dataset.test_evaluation \
        if target_dataset is not None else None

    # Information about domains
    num_domains = 2  # we'll always assume 2 domains
    num_classes = source_dataset.num_classes

    # Loss functions
    task_loss = models.make_task_loss(FLAGS.adapt)
    domain_loss = models.make_domain_loss(FLAGS.adapt)

    # Source domain will be [[1,0], [1,0], ...] and target domain [[0,1], [0,1], ...]
    source_domain = domain_labels(0, train_batch, num_domains)
    target_domain = domain_labels(1, train_batch, num_domains)

    # We need to know where we are in training for the GRL lambda schedule
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Build our model
    model = models.DomainAdaptationModel(num_classes, num_domains, FLAGS.model,
        global_step, FLAGS.steps)

    # Optimizers
    opt = tf.keras.optimizers.Adam(FLAGS.lr)
    d_opt = tf.keras.optimizers.Adam(FLAGS.lr*FLAGS.lr_mult)

    # Checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step, opt=opt, d_opt=d_opt, model=model)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir)
    checkpoint_manager.restore_latest()

    # Metrics
    have_target_domain = target_dataset is not None
    metrics = Metrics(log_dir, source_dataset, num_domains,
        task_loss, domain_loss, have_target_domain)

    # Start training
    for i in range(int(global_step), FLAGS.steps+1):
        # Get data for this iteration
        data_a = next(source_iter)
        data_b = next(target_iter) if target_iter is not None else None

        t = time.time()
        train_step(data_a, data_b, model, opt, d_opt,
            source_domain, target_domain, task_loss, domain_loss)
        global_step.assign_add(1)
        t = time.time() - t

        if i%10 == 0:
            logging.info("step %d took %f seconds", int(global_step), t)

        # Metrics on training/validation data
        if i%FLAGS.log_train_steps == 0:
            metrics.train(model, data_a, data_b, global_step, t)

        validation_accuracy = None
        if i%FLAGS.log_val_steps == 0:
            validation_accuracy = metrics.test(model,
                source_dataset_eval, target_dataset_eval, global_step)

        # Checkpoints -- Save either if at the right model step or if we found
        # a new validation accuracy. If this is better than the previous best
        # model, we need to make a new checkpoint so we can restore from this
        # step with the best accuracy.
        if i%FLAGS.model_steps == 0 or validation_accuracy is not None:
            checkpoint_manager.save(int(global_step-1), validation_accuracy)

    # We're done -- used for hyperparameter tuning
    write_finished(log_dir)


if __name__ == "__main__":
    app.run(main)
