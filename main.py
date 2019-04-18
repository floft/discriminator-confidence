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
from file_utils import last_modified_number, write_finished

FLAGS = flags.FLAGS

flags.DEFINE_enum("model", None, models.names(), "What model type to use")
flags.DEFINE_string("modeldir", "models", "Directory for saving model files")
flags.DEFINE_string("logdir", "logs", "Directory for saving log files")
flags.DEFINE_enum("method", None, ["none", "dann", "att", "pseudo"], "What method of domain adaptation to perform (or none)")
flags.DEFINE_enum("source", None, load_datasets.names(), "What dataset to use as the source")
flags.DEFINE_enum("target", "", [""]+load_datasets.names(), "What dataset to use as the target")
flags.DEFINE_integer("steps", 80000, "Number of training steps to run")
flags.DEFINE_float("lr", 0.001, "Learning rate for training")
flags.DEFINE_float("lr_domain_mult", 1.0, "Learning rate multiplier for training domain classifier")
flags.DEFINE_float("lr_target_mult", 1.0, "Learning rate multiplier for training target classifier")
flags.DEFINE_float("lr_pseudo_mult", 0.5, "Learning rate multiplier for training task classifier on pseudo-labeled data")
flags.DEFINE_float("gpumem", 0.3, "Percentage of GPU memory to let TensorFlow use")
flags.DEFINE_integer("model_steps", 4000, "Save the model every so many steps")
flags.DEFINE_integer("log_train_steps", 500, "Log training information every so many steps")
flags.DEFINE_integer("log_val_steps", 4000, "Log validation information every so many steps (also saves model)")
flags.DEFINE_boolean("target_classifier", True, "Use separate target classifier in ATT or Pseudo[-labeling] methods")
flags.DEFINE_boolean("use_grl", False, "Use gradient reversal layer for training discriminator for adaptation")
flags.DEFINE_boolean("test", False, "Use real test set for evaluation rather than validation set")
flags.DEFINE_boolean("debug", False, "Start new log/model/images rather than continuing from previous run")
flags.DEFINE_integer("debugnum", -1, "Specify exact log/model/images number to use rather than incrementing from last. (Don't pass both this and --debug at the same time.)")

flags.mark_flag_as_required("model")
flags.mark_flag_as_required("method")
flags.mark_flag_as_required("source")


def get_directory_names():
    """ Figure out the log and model directory names """
    prefix = FLAGS.source+"-"+FLAGS.target+"-"+FLAGS.model

    methods_suffix = {
        "none": "",
        "dann": "-dann",
        "att": "-att",
        "pseudo": "-pseudo",
    }

    prefix += methods_suffix[FLAGS.method]

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
def train_step_grl(data_a, data_b, model, opt, d_opt,
        task_loss, domain_loss):
    """ Compiled DANN (with GRL) training step that we call many times """
    x_a, y_a = data_a
    x_b, y_b = data_b

    # Concatenate for adaptation - concatenate source labels with all-zero
    # labels for target since we can't use the target labels during
    # unsupervised domain adaptation
    x = tf.concat((x_a, x_b), axis=0)
    task_y_true = tf.concat((y_a, tf.zeros_like(y_b)), axis=0)

    half_batch_size = tf.shape(x)[0] / 2
    source_domain = tf.zeros([half_batch_size, 1])
    target_domain = tf.ones([half_batch_size, 1])
    domain_y_true = tf.concat((source_domain, target_domain), axis=0)

    with tf.GradientTape() as tape, tf.GradientTape() as d_tape:
        task_y_pred, domain_y_pred = model(x, training=True)
        d_loss = domain_loss(domain_y_true, domain_y_pred)
        loss = task_loss(task_y_true, task_y_pred, training=True) + d_loss

    grad = tape.gradient(loss, model.trainable_variables_task_domain)
    opt.apply_gradients(zip(grad, model.trainable_variables_task_domain))

    # Update discriminator again
    d_grad = d_tape.gradient(d_loss, model.domain_classifier.trainable_variables)
    d_opt.apply_gradients(zip(d_grad, model.domain_classifier.trainable_variables))


@tf.function
def train_step_gan(data_a, data_b, model, opt, d_opt,
        task_loss, domain_loss):
    """ Compiled multi-step (GAN-like, see Shu et al. VADA paper) adaptation
    training step that we call many times

    Feed through separately so we get different batch normalizations for each
    domain. Also optimize in a GAN-like manner rather than with GRL."""
    x_a, y_a = data_a
    x_b, y_b = data_b

    # The VADA "replacing gradient reversal" (note D(f(x)) = probability of
    # being target) with non-saturating GAN-style training
    #with tf.GradientTape() as t_tape, \
    #        tf.GradientTape() as f_tape, \
    #        tf.GradientTape() as d_tape:
    with tf.GradientTape(persistent=True) as tape:
        task_y_pred_a, domain_y_pred_a = model(x_a, training=True)
        _, domain_y_pred_b = model(x_b, training=True)

        # Correct task labels (only for source domain)
        task_y_true_a = y_a

        # Correct domain labels
        # Source domain = 0, target domain = 1
        domain_y_true_a = tf.zeros_like(domain_y_pred_a)
        domain_y_true_b = tf.ones_like(domain_y_pred_b)

        # Update feature extractor and task classifier to correctly predict
        # labels on source domain
        t_loss = task_loss(task_y_true_a, task_y_pred_a)

        # Update feature extractor to fool discriminator - min_theta step
        # (swap ones/zeros from correct, update FE rather than D weights)
        d_loss_fool = domain_loss(domain_y_true_b, domain_y_pred_a) \
            + domain_loss(domain_y_true_a, domain_y_pred_b)

        # Update discriminator - min_D step
        # (train D to be correct, update D weights)
        d_loss_true = domain_loss(domain_y_true_a, domain_y_pred_a) \
            + domain_loss(domain_y_true_b, domain_y_pred_b)

    fe_and_task_variables = model.feature_extractor.trainable_variables \
        + model.task_classifier.trainable_variables

    #t_grad = t_tape.gradient(t_loss, fe_and_task_variables)
    #f_grad = f_tape.gradient(d_loss_fool, model.feature_extractor.trainable_variables)
    #d_grad = d_tape.gradient(d_loss_true, model.domain_classifier.trainable_variables)

    t_grad = tape.gradient(t_loss, fe_and_task_variables)
    f_grad = tape.gradient(d_loss_fool, model.feature_extractor.trainable_variables)
    d_grad = tape.gradient(d_loss_true, model.domain_classifier.trainable_variables)
    del tape

    # TODO maybe separate for these?
    t_opt = opt
    f_opt = opt

    t_opt.apply_gradients(zip(t_grad, fe_and_task_variables))
    f_opt.apply_gradients(zip(f_grad, model.feature_extractor.trainable_variables))
    d_opt.apply_gradients(zip(d_grad, model.domain_classifier.trainable_variables))

    # TODO for inference, use the exponential moving average of the batch norm
    # statistics on the *target* data -- above will mix them probably.

    # TODO for inference use exponential moving average of *weights* (see VADA)


@tf.function
def train_step_none(data_a, data_b, model, opt, d_opt,
        task_loss, domain_loss):
    """ Compiled no adaptation training step that we call many times """
    x_a, y_a = data_a

    with tf.GradientTape() as tape:
        task_y_pred, _ = model(x_a, training=True)
        task_y_true = y_a
        loss = task_loss(task_y_true, task_y_pred, training=True)

    grad = tape.gradient(loss, model.trainable_variables_task)
    opt.apply_gradients(zip(grad, model.trainable_variables_task))


@tf.function
def pseudo_label(x, model):
    """ Compiled step for pseudo-labeling target data """
    # Run target data through model, return the predictions and probability
    # of being source data
    # TODO also possible to weight updates to normal "task classifier" by the
    # probability the data is *target* data (opposite of this, where here we
    # look for data that it thinks is *source* data)
    task_y_pred, domain_y_pred = model(x, training=True)

    # TODO output from domain classifier is now a logit, so needs to be passed
    # through sigmoid before using as a weight. Also, now it's only one value:
    # the probability of being target (not source or target).

    batch_size = tf.shape(domain_y_pred)[0]
    domain_prob_source = tf.slice(domain_y_pred, [0, 0], [batch_size, 1])
    domain_prob_target = tf.slice(domain_y_pred, [0, 1], [batch_size, 1])

    return task_y_pred, domain_prob_source, domain_prob_target


@tf.function
def train_step_target(data_b, weights, model, opt, weighted_task_loss,
        target_classifier):
    """ Compiled train step for pseudo-labeled target data """
    x, task_y_pseudo = data_b

    # Run data through model and compute loss
    with tf.GradientTape() as tape:
        task_y_pred, domain_y_pred = model(x, target=target_classifier, training=True)
        loss = weighted_task_loss(task_y_pseudo, task_y_pred, weights, training=True)

    # Only update feature extractor and target classifier
    if target_classifier:
        trainable_vars = model.trainable_variables_target
    else:
        trainable_vars = model.trainable_variables_task

    # Update model
    grad = tape.gradient(loss, trainable_vars)
    opt.apply_gradients(zip(grad, trainable_vars))


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

    # We adapt for any of the methods other than "none" (no adaptation)
    adapt = FLAGS.method != "none"

    # For adaptation, we'll be concatenating together half source and half target
    # data, so to keep the batch_size about the same, we'll cut it in half
    train_batch = FLAGS.train_batch

    if adapt and FLAGS.use_grl:
        train_batch = train_batch // 2

    # Input training data
    #
    # Note: "It is worth noting that only the training sets of the small image
    # datasets were used during training; the test sets used for reporting
    # scores only." (self-ensembling) -- so, only use *_test for evaluation.
    # However, for now we'll use 1000 random target test samples for the
    # validation dataset (as is common).
    if FLAGS.target != "":
        source_dataset, target_dataset = load_datasets.load_da(FLAGS.source,
            FLAGS.target, test=FLAGS.test, train_batch=train_batch)
        assert source_dataset.num_classes == target_dataset.num_classes, \
            "Adapting from source to target with different classes not supported"
    else:
        raise NotImplementedError("currently don't support only source")
        source_dataset = load_datasets.load(FLAGS.source, test=FLAGS.test,
            train_batch=train_batch)
        target_dataset = None

    # Iterator and evaluation datasets if we have the dataset
    source_iter = iter(source_dataset.train)
    source_dataset_eval = source_dataset.test_evaluation
    target_iter = iter(target_dataset.train) \
        if target_dataset is not None else None
    target_dataset_eval = target_dataset.test_evaluation \
        if target_dataset is not None else None

    # Information about domains
    num_classes = source_dataset.num_classes

    # Loss functions
    task_loss = models.make_task_loss(adapt and FLAGS.use_grl)
    domain_loss = models.make_domain_loss(adapt)
    weighted_task_loss = models.make_weighted_loss()

    # We need to know where we are in training for the GRL lambda schedule
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Build our model
    model = models.DomainAdaptationModel(num_classes, FLAGS.model,
        global_step, FLAGS.steps, use_grl=FLAGS.use_grl)

    # Optimizers
    opt = tf.keras.optimizers.Adam(FLAGS.lr)
    d_opt = tf.keras.optimizers.Adam(FLAGS.lr*FLAGS.lr_domain_mult)

    # Target classifier optimizer if target_classifier, otherwise the optimizer
    # for the task-classifier when running on pseudo-labeled data
    do_pseudo_labeling = FLAGS.method in ["att", "pseudo"]
    has_target_classifier = do_pseudo_labeling and FLAGS.target_classifier
    t_mult = FLAGS.lr_target_mult if has_target_classifier else FLAGS.lr_pseudo_mult
    t_opt = tf.keras.optimizers.Adam(FLAGS.lr*t_mult)

    # Checkpoints
    checkpoint = tf.train.Checkpoint(
        global_step=global_step, opt=opt, d_opt=d_opt, t_opt=t_opt, model=model)
    checkpoint_manager = CheckpointManager(checkpoint, model_dir, log_dir,
        target=has_target_classifier)
    checkpoint_manager.restore_latest()

    # Metrics
    has_target_domain = target_dataset is not None
    metrics = Metrics(log_dir, source_dataset,
        task_loss, domain_loss, has_target_domain, has_target_classifier)

    # Start training
    for i in range(int(global_step), FLAGS.steps+1):
        # Get data for this iteration
        data_a = next(source_iter)
        data_b = next(target_iter) if target_iter is not None else None

        t = time.time()
        step_args = (data_a, data_b, model, opt, d_opt, task_loss, domain_loss)

        if adapt and FLAGS.use_grl:
            train_step_grl(*step_args)
        elif adapt:
            train_step_gan(*step_args)
        else:
            train_step_none(*step_args)

        if do_pseudo_labeling:
            # We'll ignore the real labels, so just get the data
            x, _ = data_b

            # Pseudo-label target data
            task_y_pred, domain_prob_source, _ = pseudo_label(x, model)

            # Create new data with same input by pseudo-labels not true labels
            data_b_pseudo = (x, task_y_pred)

            # Train target classifier on pseudo-labeled data, weighted
            # by probability that it's source data (i.e. higher confidence)
            train_step_target(data_b_pseudo, domain_prob_source, model,
                t_opt, weighted_task_loss, has_target_classifier)

        global_step.assign_add(1)
        t = time.time() - t

        if i%10 == 0:
            logging.info("step %d took %f seconds", int(global_step), t)

        # Metrics on training/validation data
        if i%FLAGS.log_train_steps == 0:
            metrics.train(model, data_a, data_b, global_step, t)

        validation_accuracy = None
        if i%FLAGS.log_val_steps == 0:
            validation_accuracy, target_validation_accuracy = metrics.test(
                model, source_dataset_eval, target_dataset_eval, global_step)

        # Checkpoints -- Save either if at the right model step or if we found
        # a new validation accuracy. If this is better than the previous best
        # model, we need to make a new checkpoint so we can restore from this
        # step with the best accuracy.
        if i%FLAGS.model_steps == 0 or validation_accuracy is not None:
            checkpoint_manager.save(int(global_step-1), validation_accuracy,
                target_validation_accuracy)

    # We're done -- used for hyperparameter tuning
    write_finished(log_dir)


if __name__ == "__main__":
    app.run(main)
