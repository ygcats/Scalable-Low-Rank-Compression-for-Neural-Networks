import tensorflow as tf
import re
from dataset.img_processing import per_channel_standardization, random_crop_with_padding, load

# Global constants
DATA_FORMAT = 'channels_first'


def parse_func_cifar(nr_classes):
    IMAGE_WIDTH = 32
    IMAGE_HEIGHT = 32
    IMAGE_CHANNEL = 3
    NUM_CLASSES = nr_classes
    PADDING_SIZE = 4
    PADDING_VALUE = 0
    MEAN = [0.4914009, 0.48215896, 0.4465308] if nr_classes == 10 else [0.5071, 0.4865, 0.4409]
    STD = [0.24703303, 0.24348447, 0.2615878] if nr_classes == 10 else [0.2673, 0.2564, 0.2762]

    def parse_function_train(filename, label):
        with tf.name_scope('train_transforms'):
            image = load(filename, IMAGE_CHANNEL)
            image = random_crop_with_padding(
                image, crop_size=[IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], pad=PADDING_SIZE, value=PADDING_VALUE
            )
            image = tf.image.random_flip_left_right(image)
            image = per_channel_standardization(image, MEAN, STD)
        return tf.transpose(image, perm=[2, 0, 1]) if DATA_FORMAT == 'channels_first' else image, \
               tf.one_hot(label, depth=NUM_CLASSES)

    def parse_function_eval(filename, label):
        with tf.name_scope('valid_transforms'):
            image = load(filename, IMAGE_CHANNEL)
            image = per_channel_standardization(image, MEAN, STD)
        return tf.transpose(image, perm=[2, 0, 1]) if DATA_FORMAT == 'channels_first' else image, \
               tf.one_hot(label, depth=NUM_CLASSES)

    return parse_function_train, parse_function_eval


def build_input_pipeline_train(
        parse_function, filenames, labels, batch_size, num_cpu_threads, num_workers=1, worker_id=0, prefetch=1,
        drop_remainder=False):
    return tf.data.Dataset.from_tensor_slices((filenames, labels)) \
        .shard(num_workers, worker_id) \
        .shuffle(buffer_size=len(filenames)) \
        .apply(tf.data.experimental.map_and_batch(
            map_func=parse_function, num_parallel_calls=num_cpu_threads,
            batch_size=batch_size // num_workers, drop_remainder=drop_remainder))\
        .prefetch(buffer_size=prefetch)


def build_input_pipeline_eval(
        parse_function, filenames, labels, batch_size, num_cpu_threads, num_workers=1, worker_id=0, prefetch=1):
    return tf.data.Dataset.from_tensor_slices((filenames, labels)) \
        .shard(num_workers, worker_id) \
        .apply(tf.data.experimental.map_and_batch(
            map_func=parse_function, num_parallel_calls=num_cpu_threads,
            batch_size=batch_size // num_workers, drop_remainder=False))\
        .prefetch(buffer_size=prefetch)


def read_path_label(filename):
    path, label = [], []
    try:
        with open(filename, mode='r') as f:
            for line in f:
                line = line.strip()     # delete ' ' and '\n'
                if line == '':
                    continue
                line = re.split('[\s,]+', line)    # split path and label
                if len(line) < 2:
                    continue
                path.append(line[0])
                label.append(int(line[1]))
    except FileNotFoundError:
        print('[read_path_label] A file: "%s" does not exist.' % (filename))
    return path, label

