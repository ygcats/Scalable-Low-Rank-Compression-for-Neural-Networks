import tensorflow as tf


def per_channel_standardization(image, mean, std, name=None):
    image = tf.divide(
        tf.subtract(image, tf.constant(mean, shape=[1, 1, 3])),
        tf.constant(std, shape=[1, 1, 3]),
        name=name
    )
    return image


def random_crop_with_padding(image, crop_size, pad=0, value=0, name=None):
    image = tf.image.random_crop(
        tf.pad(image, paddings=[[pad, pad], [pad, pad], [0, 0]], constant_values=value),
        size=crop_size,
        name=name
    )
    return image


def load(filename, nr_channels):
    image_string = tf.read_file(filename)
    image = tf.image.decode_image(image_string, channels=nr_channels)
    image.set_shape([None, None, nr_channels])  # Set static shape for building the graph
    image = tf.cast(image, tf.float32) / 255.   # This will convert to float values in [0, 1]
    return image

