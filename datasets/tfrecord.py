"""
Functions to write the x,y data to a tfrecord file
"""
import tensorflow as tf


def _bytes_feature(value):
    """ Returns a bytes_list from a string / byte. """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tf_example(x, y):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'x': _bytes_feature(tf.io.serialize_tensor(x)),
        'y': _bytes_feature(tf.io.serialize_tensor(y)),
    }))
    return tf_example


def write_tfrecord(filename, x, y):
    """ Output to TF record file """
    assert len(x) == len(y)
    options = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.GZIP)

    with tf.io.TFRecordWriter(filename, options=options) as writer:
        for i in range(len(x)):
            tf_example = create_tf_example(x[i], y[i])
            writer.write(tf_example.SerializeToString())


def tfrecord_filename(domain1, domain2, dataset_name, train_or_test):
    """
    Determine tfrecord filename for source --> target adaptation,
    loading the dataset_name (one of source or target) for training or testing
    """
    names = [domain1, domain2]

    # Sanity checks
    assert train_or_test in ["train", "test"], \
        "train_or_test must be one of \"train\" or \"test\""
    assert dataset_name in names, \
        "dataset_name must be one of domain1 or domain2"

    # Prefix is the source and target names but sorted
    names.sort()
    prefix = names[0]+"_and_"+names[1]

    filename = "%s_%s_%s.tfrecord"%(prefix, dataset_name, train_or_test)

    return filename
