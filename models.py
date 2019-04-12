"""
Models

Primarily we can't be using just sequential because of the grl_lambda needing
to be passed around. Also residual connections need to be a separate custom
layer.

Provides the model DomainAdaptationModel() and its components along with the
make_{task,domain}_loss() functions. Also, compute_accuracy() if desired.
"""
import tensorflow as tf

from absl import flags
from tensorflow.python.keras import backend as K

FLAGS = flags.FLAGS

flags.DEFINE_float("dropout", 0.05, "Dropout probability")

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

    def call(self, inputs, **kwargs):
        """ Calculate grl_lambda first based on the current global step (a
        variable) and then create the layer that does nothing except flip
        the gradients """
        grl_lambda = self.grl_schedule(self.global_step)
        return flip_gradient(inputs, grl_lambda)


class StopGradient(tf.keras.layers.Layer):
    """ Stop gradient layer """
    def call(self, inputs, **kwargs):
        return tf.stop_gradient(inputs)


def make_dense_bn_dropout(units, dropout):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.Dropout(dropout),
    ])


class ResnetBlock(tf.keras.layers.Layer):
    """ Block consisting of other blocks but with residual connections """
    def __init__(self, units, dropout, layers, **kwargs):
        super().__init__(**kwargs)
        self.blocks = [make_dense_bn_dropout(units, dropout) for _ in range(layers)]
        self.add = tf.keras.layers.Add()

    def call(self, inputs, **kwargs):
        """ Like Sequential but with a residual connection """
        shortcut = inputs
        net = inputs

        for block in self.blocks:
            net = block(net, **kwargs)

        return self.add([shortcut, net], **kwargs)


def DannGrlSchedule(num_steps):
    """ GRL schedule from DANN paper """
    def schedule(step):
        return 2/(1+tf.exp(-10*(step/(num_steps+1))))-1
    return schedule


def make_vrada_model(num_classes, num_domains, global_step, grl_schedule):
    """
    Create model inspired by the VRADA paper model for time-series data

    Note: VRADA model had a VRNN though rather than just flattening data and
    didn't use residual connections.
    """
    fe_layers = 5
    task_layers = 1
    domain_layers = 2
    resnet_layers = 2
    units = 50
    dropout = FLAGS.dropout

    # General classifier used in both the task/domain classifiers
    def make_classifier(layers, num_outputs):
        layers = [make_dense_bn_dropout(units, dropout) for _ in range(layers-1)]
    last = [
            tf.keras.layers.Dense(num_outputs),
        tf.keras.layers.Activation("softmax"),
    ]
    return tf.keras.Sequential(layers + last)

    feature_extractor = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(momentum=0.999),
    ] + [  # First can't be residual since x isn't of size units
        make_dense_bn_dropout(units, dropout) for _ in range(resnet_layers)
    ] + [
        ResnetBlock(units, dropout, resnet_layers) for _ in range(fe_layers-1)
    ])
    task_classifier = tf.keras.Sequential([
        make_classifier(task_layers, num_classes),
    ])
    domain_classifier = tf.keras.Sequential([
        FlipGradient(global_step, grl_schedule),
        make_classifier(domain_layers, num_domains),
    ])
    return feature_extractor, task_classifier, domain_classifier


class DomainAdaptationModel(tf.keras.Model):
    """
    Domain adaptation model -- task and domain classifier outputs, depends on
    command line --model=X argument

    Usage:
        model = DomainAdaptationModel(num_classes, num_domains, "flat",
            global_step, num_steps)

        with tf.GradientTape() as tape:
            task_y_pred, domain_y_pred = model(x, training=True)
            ...
    """
    def __init__(self, num_classes, num_domains, model_name, global_step,
            num_steps, **kwargs):
        super().__init__(**kwargs)
        grl_schedule = DannGrlSchedule(num_steps)

        if model_name == "flat":
            fe, task, domain = make_vrada_model(num_classes, num_domains,
                global_step, grl_schedule)
        else:
            raise NotImplementedError("Model name: "+str(model_name))

        self.feature_extractor = fe
        self.task_classifier = task
        self.domain_classifier = domain

    @property
    def trainable_variables_exclude_domain(self):
        """ Same as .trainable_variables but excluding the domain classifier """
        return self.feature_extractor.trainable_variables \
            + self.task_classifier.trainable_variables

    def call(self, inputs, training=None, **kwargs):
        # Manually set the learning phase since we probably aren't using .fit()
        if training is True:
            tf.keras.backend.set_learning_phase(1)
        elif training is False:
            tf.keras.backend.set_learning_phase(0)

        fe = self.feature_extractor(inputs, **kwargs)
        task = self.task_classifier(fe, **kwargs)
        domain = self.domain_classifier(fe, **kwargs)
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


# List of names
models = [
    "flat",
    "dann_mnist",
    "dann_svhn",
    "dann_gtsrb",
]


# Get names
def names():
    """
    Returns list of all the available models for use in DomainAdaptationModel()
    """
    return models
