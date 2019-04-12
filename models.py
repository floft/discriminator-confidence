"""
Models

Primarily we can't be using just sequential because of the grl_lambda needing
to be passed around. Also residual connections need to be a separate custom
layer.

Provides the model DomainAdaptationModel() and its components along with the
make_{task,domain}_loss() functions. Also, compute_accuracy() if desired.

Usage:
    # Build our model
    model = DomainAdaptationModel(num_classes, num_domains, name_of_model)

    # During training
    task_y_pred, domain_y_pred = model(x, grl_lambda=1.0, training=True)

    # During evaluation
    task_y_pred, domain_y_pred = model(x, training=False)
"""
import tensorflow as tf

from absl import flags
from tensorflow.python.keras import backend as K

FLAGS = flags.FLAGS

flags.DEFINE_float("dropout", 0.05, "Dropout probability")
flags.DEFINE_integer("units", 50, "Number of LSTM hidden units and VRNN latent variable size")
flags.DEFINE_integer("layers", 5, "Number of layers for the feature extractor")
flags.DEFINE_integer("task_layers", 1, "Number of layers for the task classifier")
flags.DEFINE_integer("domain_layers", 2, "Number of layers for the domain classifier")
flags.DEFINE_integer("resnet_layers", 2, "Number of layers within a single resnet block")

flags.register_validator("dropout", lambda v: v != 1, message="dropout cannot be 1")


    @tf.custom_gradient
    def flip_gradient(x, grl_lambda=1.0):
    """ Forward pass identity, backward pass negate gradient and multiply by  """
        grl_lambda = tf.cast(grl_lambda, dtype=tf.float32)

        def grad(dy):
        return tf.negative(dy) * grl_lambda * tf.ones_like(x)

        return x, grad


class FlipGradient(tf.keras.layers.Layer):
    """
    Gradient reversal layer

    global_step = tf.Variable storing the current step
    schedule = a function taking the global_step and computing the grl_lambda,
        e.g. `lambda step: 1.0` or some more complex function.
    """
    def __init__(self, global_step, grl_schedule, **kwargs):
        super().__init__(**kwargs)
        self.global_step = global_step
        self.grl_schedule = grl_schedule

    def call(self, inputs, training=None):
        """ Calculate grl_lambda first based on the current global step (a
        variable) and then create the layer that does nothing except flip
        the gradients """
        grl_lambda = self.grl_schedule(self.global_step)
        return flip_gradient(inputs, grl_lambda)


class StopGradient(tf.keras.layers.Layer):
    """ Stop gradient layer """
    def call(self, inputs, training=None):
        return tf.stop_gradient(inputs)


class DenseBlock(tf.keras.layers.Layer):
    """
    Dense block with batch norm and dropout

    Note: doing this rather than Sequential because dense gives error if we pass
    training=True to it
    """
    def __init__(self, units, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(units)
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation("relu")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=None):
        net = self.dense(inputs)
        net = self.bn(net, training=training)
        net = self.act(net)
        net = self.dropout(net, training=training)
        return net


class ResnetBlock(tf.keras.layers.Layer):
    """ Block consisting of other blocks but with residual connections """
    def __init__(self, units, dropout, layers=None,
            make_block=DenseBlock, **kwargs):
        super().__init__(**kwargs)

        if layers is None:
            layers = FLAGS.resnet_layers

        self.blocks = [make_block(units, dropout) for _ in range(layers)]
        self.add = tf.keras.layers.Add()

    def call(self, inputs, training=None):
        """ Like Sequential but with a residual connection """
        shortcut = inputs
        net = inputs

        for block in self.blocks:
            net = block(net, training=training)

        return self.add([shortcut, net])


class Classifier(tf.keras.layers.Layer):
    """ MLP classifier -- multiple DenseBlock followed by dense of size
    num_classes and softmax """
    def __init__(self, layers, units, dropout, num_classes,
            make_block=DenseBlock, **kwargs):
        super().__init__(**kwargs)
        assert layers > 0, "must have layers > 0"
        self.blocks = [make_block(units, dropout) for _ in range(layers-1)]
        self.dense = tf.keras.layers.Dense(num_classes)
        self.act = tf.keras.layers.Activation("softmax")

    def call(self, inputs, training=None):
        net = inputs

        for block in self.blocks:
            net = block(net, training=training)

        net = self.dense(net)
        net = self.act(net)

        return net


class DomainClassifier(tf.keras.layers.Layer):
    """ Classifier() but flipping gradients """
    def __init__(self, layers, units, dropout, num_domains,
            global_step, grl_schedule,
            make_classifier=Classifier, **kwargs):
        super().__init__(**kwargs)
        self.flip_gradient = FlipGradient(global_step, grl_schedule)
        self.classifier = make_classifier(layers, units, dropout, num_domains)

    def call(self, inputs, training=None):
        net = self.flip_gradient(inputs, training=training)
        net = self.classifier(net, training=training)
        return net


class FeatureExtractor(tf.keras.layers.Layer):
    """ Resnet feature extractor """
    def __init__(self, layers, units, dropout,
            make_base_block=DenseBlock, make_res_block=ResnetBlock, **kwargs):
        super().__init__(**kwargs)
        assert layers > 0, "must have layers > 0"
        # First can't be residual since x isn't of size units
        self.blocks = [make_base_block(units, dropout) for _ in range(FLAGS.resnet_layers)]
        self.blocks += [make_res_block(units, dropout) for _ in range(layers-1)]

    def call(self, inputs, training=None):
        net = inputs

        for block in self.blocks:
            net = block(net, training=training)

        return net


class FlatModel(tf.keras.layers.Layer):
    """ Flatten and normalize then model """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.999)

    def call(self, inputs, training=None):
        net = self.flatten(inputs)
        return self.bn(net, training=training)


def DannGrlSchedule(num_steps):
    """ GRL schedule from DANN paper """
    def schedule(step):
        return 2/(1+tf.exp(-10*(step/(num_steps+1))))-1
    return schedule


class DomainAdaptationModel(tf.keras.Model):
    """
    Contains custom model, feature extractor, task classifier, and domain
    classifier

    The custom model before the feature extractor depends on the command line
    argument.

    Usage:
        model = DomainAdaptationModel(num_classes, num_domains, "flat")

        with tf.GradientTape() as tape:
            task_y_pred, domain_y_pred = model(x, grl_lambda=1.0, training=True)
            ...
    """
    def __init__(self, num_classes, num_domains, model_name, global_step,
            num_steps, **kwargs):
        super().__init__(**kwargs)

        grl_schedule = DannGrlSchedule(num_steps)

        self.feature_extractor = FeatureExtractor(FLAGS.layers, FLAGS.units, FLAGS.dropout)
        self.task_classifier = Classifier(FLAGS.task_layers, FLAGS.units,
            FLAGS.dropout, num_classes)
        self.domain_classifier = DomainClassifier(FLAGS.domain_layers, FLAGS.units,
            FLAGS.dropout, num_domains, global_step, grl_schedule)

        if model_name == "flat":
            self.custom_model = FlatModel()
        else:
            raise NotImplementedError("Model name: "+str(model_name))

    @property
    def trainable_variables_exclude_domain(self):
        """ Same as .trainable_variables but excluding the domain classifier """
        return self.feature_extractor.trainable_variables \
            + self.task_classifier.trainable_variables \
            + self.custom_model.trainable_variables

    def call(self, inputs, training=None):
        net = self.custom_model(inputs, training=training)
        net = self.feature_extractor(net, training=training)
        task = self.task_classifier(net, training=training)
        domain = self.domain_classifier(net, training=training)
        return task, domain


def make_task_loss(adapt):
    """
    The same as CategoricalCrossentropy() but only on half the batch if doing
    adaptation and in the training phase
    """
    cce = tf.keras.losses.CategoricalCrossentropy()

    def task_loss(y_true, y_pred, training=None):
        """
        Compute loss on the outputs of the task classifier

        Note: domain classifier can use normal tf.keras.losses.CategoricalCrossentropy
        but for the task loss when doing adaptation we need to ignore the second half
        of the batch since this is unsupervised
        """
        if training is None:
            training = K.learning_phase()

        # If doing domain adaptation, then we'll need to ignore the second half of the
        # batch for task classification during training since we don't know the labels
        # of the target data
        if adapt and training:
            batch_size = tf.shape(y_pred)[0]
            y_pred = tf.slice(y_pred, [0, 0], [batch_size // 2, -1])
            y_true = tf.slice(y_true, [0, 0], [batch_size // 2, -1])

        return cce(y_true, y_pred)

    return task_loss


def make_domain_loss(use_domain_loss):
    """
    Just CategoricalCrossentropy() but for consistency with make_task_loss()
    """
    if use_domain_loss:
        cce = tf.keras.losses.CategoricalCrossentropy()

        def domain_loss(y_true, y_pred):
            """ Compute loss on the outputs of the domain classifier """
            return cce(y_true, y_pred)
    else:
        def domain_loss(y_true, y_pred):
            """ Domain loss only used during adaptation """
            return 0

    return domain_loss


def compute_accuracy(y_true, y_pred):
    return tf.reduce_mean(input_tensor=tf.cast(
        tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)),
        tf.float32))
