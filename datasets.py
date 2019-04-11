"""
Datasets

Usage:
    import datasets

    print(datasets.names()) # mnist, usps, svhn, ...
    mnist = datasets.load("mnist")
    class_number = mnist.label_to_int("5")
    class_name = mnist.int_to_label(5)

    # Training
    train_iter = iter(mnist.train):
    labels = mnist.class_labels
    while True:
        next_batch = next(train_iter)

    # Evaluation
    for x, y in mnist.train_evaluation:
        ...
    for x, y in mnist.test_evaluation:
        ...
"""
import gzip
import scipy.io
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("train_batch", 256, "Batch size for training")
flags.DEFINE_integer("eval_batch", 16384, "Batch size for evaluation")
flags.DEFINE_integer("shuffle_buffer", 60000, "Dataset shuffle buffer size")
flags.DEFINE_integer("prefetch_buffer", 1, "Dataset prefetch buffer size")
flags.DEFINE_integer("eval_shuffle_seed", 0, "Evaluation shuffle seed for repeatability")
flags.DEFINE_integer("eval_max_examples", 0, "Max number of examples to evaluate for validation (default 0, i.e. all)")


class Dataset:
    """
    Base class for datasets

    class Something(Dataset):
        def __init__(self, *args, **kwargs):
            num_classes = 2
            class_labels = ["class1", "class2"]
            super().__init__(num_classes, class_labels, *args, **kwargs)

        def process(self, data, labels):
            ...
            return super().process(data, labels)

        def load(self):
            ...
            return train_images, train_labels, test_images, test_labels

    Also, add to the datasets={"something": Something, ...} dictionary below.
    """
    def __init__(self, num_classes, class_labels, train_batch=None,
            eval_batch=None, shuffle_buffer=None, prefetch_buffer=None,
            eval_shuffle_seed=None, eval_max_examples=None,
            resize=None, pad_to=None, pad_const=0,
            convert_to_gray=False, convert_to_rgb=False):
        """
        Initialize dataset

        Must specify num_classes and class_labels (the names of the classes).
        Resize and pad_to should be a list of the 2D size (width, height) of
        the desired output tensor dimensions (excluding batch size and number
        of channels). Pad_const is the value to pad with. Resize is done before
        padding. Convert gray/rgb if true will convert to 1 or 3 channels
        respectively. Other arguments if None are defaults from command line
        flags.

        For example,
            Dataset(num_classes=2, class_labels=["class1", "class2"],
                resize=[28,28], pad=[32,32], pad_const=-1, convert_to_gray=True)

        This calls load() to get the data, process() to normalize, convert to
        float, etc., and then load_dataset() to create the tf.data.Dataset's.
        """
        # Sanity checks
        assert num_classes == len(class_labels), \
            "num_classes != len(class_labels)"
        assert resize is None or len(resize) == 2, \
            "Incorrect format for resize: resize=[width,height]"
        assert pad_to is None or len(pad_to) == 2, \
            "Incorrect format for pad: pad=[width,height]"
        assert not (convert_to_gray and convert_to_rgb), \
            "Cannot convert to both gray and rgb, only one or the other"

        # Set parameters
        self.num_classes = num_classes
        self.class_labels = class_labels
        self.train_batch = train_batch
        self.eval_batch = eval_batch
        self.shuffle_buffer = shuffle_buffer
        self.prefetch_buffer = prefetch_buffer
        self.eval_shuffle_seed = eval_shuffle_seed
        self.eval_max_examples = eval_max_examples
        self.resize = resize
        self.pad_to = pad_to
        self.pad_const = pad_const
        self.convert_to_gray = convert_to_gray
        self.convert_to_rgb = convert_to_rgb

        # Set defaults if not specified
        if self.train_batch is None:
            self.train_batch = FLAGS.train_batch
        if self.eval_batch is None:
            self.eval_batch = FLAGS.eval_batch
        if self.shuffle_buffer is None:
            self.shuffle_buffer = FLAGS.shuffle_buffer
        if self.prefetch_buffer is None:
            self.prefetch_buffer = FLAGS.prefetch_buffer
        if self.eval_shuffle_seed is None:
            self.eval_shuffle_seed = FLAGS.eval_shuffle_seed
        if self.eval_max_examples is None:
            self.eval_max_examples = FLAGS.eval_max_examples

        # Load the dataset
        train_images, train_labels, test_images, test_labels = self.load()
        self.train_images, self.train_labels = self.process(train_images, train_labels)
        self.test_images, self.test_labels = self.process(test_images, test_labels)
        self.train, self.train_evaluation, self.test_evaluation = \
            self.load_dataset(self.train_images, self.train_labels,
            self.test_images, self.test_labels)

    def load(self):
        raise NotImplementedError("must implement load() for Dataset class")

    def download_dataset(self, files_to_download, url, train_index=0, test_index=1):
        """
        Download url/file for file in files_to_download
        Returns: the downloaded filename at train_index, test_index (e.g. 0 and 1,
            if you passed the train filename first and test filename second).
        """
        downloaded_files = []
        for f in files_to_download:
            downloaded_files.append(tf.keras.utils.get_file(fname=f, origin=url+"/"+f))
        train_fp = downloaded_files[train_index]
        test_fp = downloaded_files[test_index]
        return train_fp, test_fp

    def tf_dataset(self, data, labels, batch_size, count=False, evaluation=False):
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))

        # If desired, take the first max_examples examples
        if evaluation and self.eval_max_examples != 0:
            dataset = dataset.take(self.eval_max_examples)

        if count:  # only count, so no need to shuffle
            pass
        elif evaluation:  # don't repeat since we want to evaluate entire set
            dataset = dataset.shuffle(self.shuffle_buffer, seed=self.eval_shuffle_seed)
        else:  # repeat, shuffle, and batch
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(self.shuffle_buffer))

        dataset = dataset.batch(batch_size).prefetch(self.prefetch_buffer)

        return dataset

    def load_dataset(self, train_images, train_labels, test_images, test_labels):
        """
        Load the X dataset as a tf.data.Dataset from train/test images/labels
        """
        train_dataset = self.tf_dataset(train_images, train_labels,
            self.train_batch)
        eval_train_dataset = self.tf_dataset(train_images, train_labels,
            self.eval_batch, evaluation=True)
        eval_test_dataset = self.tf_dataset(test_images, test_labels,
            self.eval_batch, evaluation=True)
        return train_dataset, eval_train_dataset, eval_test_dataset

    def calculate_padding(self, input_size, desired_output_size):
        """ Calculate before/after padding to nearly center it, if odd then
        one more pixel of padding after rather than before """
        pad_needed = max(0, desired_output_size - input_size)
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before
        return pad_before, pad_after

    def process(self, data, labels):
        """ Perform desired resize, padding, conversions, etc. If you override,
        you should `return super().process(data, labels)` to make sure these
        options are handled. """
        if self.resize is not None:
            # Default interpolation is bilinear
            data = tf.image.resize(data, self.resize)

        if self.pad_to is not None:
            # data.shape = [batch_size, height, width, channels]
            # self.pad = [desired_width, desired_height]
            padding = [(0, 0),
                self.calculate_padding(data.shape[1], self.pad_to[1]),
                self.calculate_padding(data.shape[2], self.pad_to[0]),
                (0, 0)]
            data = tf.pad(data, padding, constant_values=self.pad_const)

        if self.convert_to_gray:
            # https://en.wikipedia.org/wiki/Luma_%28video%29
            data = tf.image.rgb_to_grayscale(data)
        elif self.convert_to_rgb:
            data = tf.image.grayscale_to_rgb(data)

        return data, labels

    def one_hot(self, y, index_one=False):
        """ One-hot encode y if not already 2D """
        squeezed = np.squeeze(y)

        if len(squeezed.shape) < 2:
            if index_one:
                y = np.eye(self.num_classes, dtype=np.float32)[squeezed.astype(np.int32) - 1]
            else:
                y = np.eye(self.num_classes, dtype=np.float32)[squeezed.astype(np.int32)]
        else:
            y = y.astype(np.float32)
            assert squeezed.shape[1] == self.num_classes, "y.shape[1] != num_classes"

        return y

    def label_to_int(self, label_name):
        """ e.g. Bathe to 0 """
        return self.class_labels.index(label_name)

    def int_to_label(self, label_index):
        """ e.g. Bathe to 0 """
        return self.class_labels[label_index]


class MNIST(Dataset):
    """ Load the MNIST dataset """
    def __init__(self, *args, **kwargs):
        num_classes = 10
        class_labels = [str(x) for x in range(10)]
        super().__init__(num_classes, class_labels, *args, **kwargs)

    def process(self, data, labels):
        """ Reshape, convert to float, normalize to [-1,1] """
        data = data.reshape(data.shape[0], 28, 28, 1).astype("float32")
        data = (data - 127.5) / 127.5
        labels = self.one_hot(labels)
        return super().process(data, labels)

    def load(self):
        (train_images, train_labels), (test_images, test_labels) = \
            tf.keras.datasets.mnist.load_data()
        return train_images, train_labels, test_images, test_labels


class SVHN(Dataset):
    """ Load the SVHN (cropped) dataset """
    def __init__(self, *args, **kwargs):
        num_classes = 10
        class_labels = [str(x) for x in range(10)]
        super().__init__(num_classes, class_labels, *args, **kwargs)

    def download(self):
        """ Download the SVHN files from online """
        train_fp, test_fp = self.download_dataset(
            ["train_32x32.mat", "test_32x32.mat", "extra_32x32.mat"],
            "http://ufldl.stanford.edu/housenumbers/")
        return train_fp, test_fp

    def process(self, data, labels):
        """ Reshape, convert to float, normalize to [-1,1] """
        data = data.reshape(data.shape[0], 32, 32, 3).astype("float32")
        data = (data - 127.5) / 127.5
        labels = self.one_hot(labels)
        return super().process(data, labels)

    def load_file(self, filename):
        """ Load from .mat file """
        data = scipy.io.loadmat(filename)
        images = data["X"].transpose([3, 0, 1, 2])
        labels = data["y"].reshape([-1])
        labels[labels == 10] = 0  # 1 = "1", 2 = "2", ... but 10 = "0"
        return images, labels

    def load(self):
        train_fp, test_fp = self.download()
        train_images, train_labels = self.load_file(train_fp)
        test_images, test_labels = self.load_file(test_fp)
        return train_images, train_labels, test_images, test_labels


class USPS(Dataset):
    """ Load the USPS dataset """
    def __init__(self, *args, **kwargs):
        num_classes = 10
        class_labels = [str(x) for x in range(10)]
        super().__init__(num_classes, class_labels, *args, **kwargs)

    def download(self):
        """ Download the USPS files from online """
        train_fp, test_fp = self.download_dataset(
            ["zip.train.gz", "zip.test.gz", "zip.info.txt"],
            "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/")
        return train_fp, test_fp

    def floats_to_list(self, line):
        """ Return list of floating-point numbers from space-separated string """
        items = line.split(" ")
        floats = []

        for item in items:
            try:
                f = float(item)
            except ValueError:
                raise ValueError("floats_to_list() requires space-separated floats")
            floats.append(f)

        return floats

    def load_file(self, filename):
        """ See zip.info.txt for the file format, which is gzipped """
        images = []
        labels = []

        with gzip.open(filename, "rb") as f:
            for line in f:
                label, *pixels = self.floats_to_list(line.strip().decode("utf-8"))
                assert len(pixels) == 256, "Should be 256 pixels"
                assert label >= 0 and label <= 9, "Label should be in [0,9]"
                images.append(pixels)
                labels.append(label)

        images = np.vstack(images)
        labels = np.hstack(labels)

        return images, labels

    def process(self, data, labels):
        """ Reshape (already normalized to [-1,1], should already be float) """
        data = data.reshape(data.shape[0], 16, 16, 1).astype("float32")
        labels = self.one_hot(labels)
        return super().process(data, labels)

    def load(self):
        train_fp, test_fp = self.download()
        train_images, train_labels = self.load_file(train_fp)
        test_images, test_labels = self.load_file(test_fp)
        return train_images, train_labels, test_images, test_labels


# List of datasets
datasets = {
    "mnist": MNIST,
    "usps": USPS,
    "svhn": SVHN,
}


# Get datasets
def load(name, *args, **kwargs):
    """ Load a dataset based on the name (must be one of datasets.names()) """
    assert name in datasets.keys(), "Name specified not in datasets.names()"
    return datasets[name](*args, **kwargs)


def load_da(source_name, target_name, *args, **kwargs):
    """ Load two datasets (source and target) but perform necessary conversions
    to make them compatable for adaptation (i.e. same size, channels, etc.).
    Names must be in datasets.names()."""
    # MNIST <-> USPS: "The USPS images were up-scaled using bilinear interpolation from
    # 16×16 to 28×28 resolution to match that of MNIST."
    if source_name == "mnist" and target_name == "usps":
        source_dataset = load(source_name, *args, **kwargs)
        target_dataset = load(target_name, *args, resize=[28, 28], **kwargs)
    elif target_name == "mnist" and source_name == "usps":
        source_dataset = load(source_name, *args, resize=[28, 28], **kwargs)
        target_dataset = load(target_name, *args, **kwargs)

    # MNIST <-> SVHN: "The MNIST images were padded to 32×32 resolution and converted
    # to RGB by replicating the greyscale channel into the three RGB channels
    # to match the format of SVHN."
    elif source_name == "mnist" and target_name == "svhn":
        source_dataset = load(source_name, *args, pad_to=[32, 32], pad_const=-1,
            convert_to_rgb=True, **kwargs)
        target_dataset = load(target_name, *args, **kwargs)
    elif target_name == "mnist" and source_name == "svhn":
        source_dataset = load(source_name, *args, **kwargs)
        target_dataset = load(target_name, *args, pad_to=[32, 32], pad_const=-1,
            convert_to_rgb=True, **kwargs)

    # No conversions, resizes, etc.
    else:
        source_dataset = load(source_name, *args, **kwargs)
        target_dataset = load(target_name, *args, **kwargs)

    return source_dataset, target_dataset


# Get names
def names():
    """
    Returns list of all the available datasets to load with datasets.load(name)
    """
    return list(datasets.keys())


def main(argv):
    print("Available datasets:", names())

    # Example showing that the sizes and number of channels are matched
    source, target = load_da("mnist", "usps")
    print("Source:", source.train)
    print("Target:", target.train)


if __name__ == "__main__":
    app.run(main)
