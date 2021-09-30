import os
import sys
import time
import random as rn

import tensorflow as tf

from dataset.img_reader import read_path_label, build_input_pipeline_eval, build_input_pipeline_train, parse_func_cifar
from common_np import full_svd_cpu, singular_value_rank
from common_tf import tf_config, count_nr_variables, compute_loss_acc, build_model, \
    tensor2mat, mat2tensor, build_low_rank_model, get_weights_for_svd, compute_params_macs, select_sorting

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('out_file_path', None, 'path to a output file')
tf.flags.DEFINE_integer('device', 0, 'gpu_id')
tf.flags.DEFINE_string('train_path', None, 'path to a txt file for training')
tf.flags.DEFINE_string('valid_path', None, 'path to a txt file for validation')
tf.flags.DEFINE_string('checkpoint', None, 'path to a checkpoint')
tf.flags.DEFINE_integer('cpu_threads', 8, '# of cpu threads')
tf.flags.DEFINE_integer('cpu_threads_np', 4, '# of cpu threads')
tf.flags.DEFINE_integer('random_seed', 0, 'random seed')
tf.flags.DEFINE_string('format', 'NCHW', 'data alignment')
tf.flags.DEFINE_string('model', 'vgg15', 'network architecture')
tf.flags.DEFINE_integer('batch_size_eval', 1024, 'size of mini-batch for evaluation step')
tf.flags.DEFINE_integer('bn_batch_num_divider', 1, '# of batch is divided by this for fast-computing of bn stats.')
tf.flags.DEFINE_list('r_ratio', [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
                                 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
                     'list of rank ratio')
tf.flags.DEFINE_integer('rank_min', 5, 'minimum rank for svd')
tf.flags.DEFINE_bool('svd_fc_last', True, 'apply svd to a matrix in the last fc layer')
tf.flags.DEFINE_string('sorting_criterion', 'sv', 'mode of sorting bases')
tf.flags.DEFINE_string('svd_decomposition', 'KyKxCin_Cout', 'mode of svd decomposition')

os.environ["OMP_NUM_THREADS"] = str(FLAGS.cpu_threads_np)         # export OMP_NUM_THREADS=FLAGS.cpu_threads
os.environ["OPENBLAS_NUM_THREADS"] = str(FLAGS.cpu_threads_np)    # export OPENBLAS_NUM_THREADS=FLAGS.cpu_threads
os.environ["MKL_NUM_THREADS"] = str(FLAGS.cpu_threads_np)         # export MKL_NUM_THREADS=FLAGS.cpu_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = str(FLAGS.cpu_threads_np)  # export VECLIB_MAXIMUM_THREADS=FLAGS.cpu_threads
os.environ["NUMEXPR_NUM_THREADS"] = str(FLAGS.cpu_threads_np)     # export NUMEXPR_NUM_THREADS=FLAGS.cpu_threads
import numpy as np
np.__config__.show()


def compute_population_mean_var(sess, init_op, nr_batches, batch_num_divider,
                                XX_T, Xl, X2l, M,
                                assign_mean, assign_var, new_mean, new_var,
                                need_cov=False):
    var = [np.zeros((x.get_shape().as_list()[0], x.get_shape().as_list()[0]), dtype='float64') if need_cov
           else np.zeros((x.get_shape().as_list()[0], 1), dtype='float64') for x in Xl]
    myu = [np.zeros((x.get_shape().as_list()[0], 1), dtype='float64') for x in Xl]
    N = [0.] * len(Xl)
    for k in range(len(Xl)):
        sess.run(init_op)
        for j in range(nr_batches // batch_num_divider):
            v, m, n = sess.run([XX_T[k], Xl[k], M[k]]) if need_cov else sess.run([X2l[k], Xl[k], M[k]])
            var[k] += v
            myu[k] += m
            N[k] += n
        myu[k] /= N[k]
        var[k] = var[k] / N[k] - np.dot(myu[k], myu[k].T) if need_cov else np.squeeze(var[k] / N[k] - np.square(myu[k]))
        myu[k] = np.squeeze(myu[k])
        sess.run([assign_mean[k], assign_var[k]],
                 feed_dict={new_mean[k]: myu[k], new_var[k]: np.diag(var[k])} if need_cov else
                           {new_mean[k]: myu[k], new_var[k]: var[k]})
    return myu, var


def get_bn_inputs(model):
    try:
        def get_layers(l):
            # check if original module or not
            if l.__class__.__module__.startswith('tensorflow.python.keras.layers'):
                return l
            else:
                # check if ListWrapper object or not
                if l.__class__.__bases__[0].__name__ == 'List':
                    return [get_layers(ll) for ll in l]
                else:
                    return [get_layers(ll) for ll in l.__dict__['_layers']]

        flatten = lambda x: [z for y in x for z in
                             (flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]

        # get keras.layers
        layers = flatten([get_layers(l) for l in model.layers])

        bn_inputs = []
        for l in layers:
            if 'BatchNormalization' in l.__class__.__name__:
                bn_inputs.append(l.input)

    except ValueError as err:
        print(err)
        return []
    return bn_inputs


def get_bn_variables(model):
    mean = [s for s in model.weights if ('moving_mean:0' in s.name)]
    var = [s for s in model.weights if ('moving_variance:0' in s.name)]
    new_mean = [tf.placeholder(shape=m.get_shape().as_list(), dtype=tf.float32) for m in mean]
    new_var = [tf.placeholder(shape=v.get_shape().as_list(), dtype=tf.float32) for v in var]
    assign_mean = [tf.assign(m, m_r) for m, m_r in zip(mean, new_mean)]
    assign_var = [tf.assign(v, v_r) for v, v_r in zip(var, new_var)]
    bn_inputs = get_bn_inputs(model)
    return mean, var, new_mean, new_var, assign_mean, assign_var, bn_inputs


def compute_per_channel_stats(inputs, format='NCHW'):
    perm = [1, 0, 2, 3] if format == 'NCHW' else [3, 0, 1, 2]
    X = [tf.transpose(x, perm=perm) if len(x.get_shape().as_list()) == 4
         else tf.transpose(x, perm=[1, 0]) for x in inputs]
    X = [tf.reshape(x, [x.get_shape().as_list()[0], -1]) for x in X]
    XX_T = [tf.matmul(x, x, transpose_b=True) for x in X]
    Xl = [tf.reduce_sum(x, 1, keepdims=True) for x in X]
    X2l = [tf.reduce_sum(tf.math.square(x), 1, keepdims=True) for x in X]
    nr_columns = [tf.shape(x)[1] for x in X]
    return XX_T, Xl, X2l, nr_columns


def main(argv):
    # parameters
    r_ratio = list(map(float, FLAGS.r_ratio))
    np.random.seed(FLAGS.random_seed)
    rn.seed(FLAGS.random_seed)
    config = tf_config(FLAGS.device)

    # load dataset
    train_paths, train_labels = read_path_label(FLAGS.train_path)
    valid_paths, valid_labels = read_path_label(FLAGS.valid_path)
    nr_labels = len(np.unique(train_labels))
    nr_samples_per_class_train = np.eye(nr_labels, dtype='float32')[train_labels].sum(axis=0)
    nr_samples_per_class_valid = np.eye(nr_labels, dtype='float32')[valid_labels].sum(axis=0)
    nr_batches_train = np.ceil(len(train_paths) / FLAGS.batch_size_eval).astype('int32')
    print('# of labels = ', nr_labels)
    print('# of training samples = ', len(train_paths))
    print('# of validation samples = ', len(valid_paths))
    print('# of training samples per class = ', nr_samples_per_class_train)
    print('# of validation samples per class = ', nr_samples_per_class_valid)
    print('# of batches for training set = ', nr_batches_train)
    print('bn_batch_num_divider = ', FLAGS.bn_batch_num_divider)
    print('rank ratio =', r_ratio)

    # make graph
    with tf.device('/cpu:0'):
        with tf.variable_scope('data'):
            # dataset
            parser_train, parser_eval = parse_func_cifar(nr_labels)
            train_set = build_input_pipeline_train(
                parser_train, train_paths, train_labels, FLAGS.batch_size_eval, FLAGS.cpu_threads)
            valid_set = build_input_pipeline_eval(
                parser_eval, valid_paths, valid_labels, FLAGS.batch_size_eval, FLAGS.cpu_threads)

            # make an iterator by data structure
            iterator = tf.data.Iterator.from_structure(
                train_set.output_types,
                ([None] + train_set.output_shapes[0].as_list()[1:],
                 [None] + train_set.output_shapes[1].as_list()[1:])
            )

            # make initializers for evaluation datasets
            train_init_op = iterator.make_initializer(train_set)
            valid_init_op = iterator.make_initializer(valid_set)

            # minibatch data
            input, target = iterator.get_next()

    with tf.device('/gpu:0'):
        tf.set_random_seed(FLAGS.random_seed)
        with tf.variable_scope('model'):
            model, _ = build_model(FLAGS.model, input, nr_labels)
            var_svd_f, var_share_f = get_weights_for_svd(model.weights, svd_fc_last=FLAGS.svd_fc_last)
            var_svd_f_mat = [tensor2mat(tensor, mode=FLAGS.svd_decomposition) for tensor in var_svd_f]

            U = [tf.placeholder(shape=[None, None], dtype=tf.float32,
                                name=os.path.join(os.path.dirname(w.name).split('/', 1)[1], 'U'))
                 for w in var_svd_f]
            S = [tf.placeholder(shape=[None], dtype=tf.float32,
                                name=os.path.join(os.path.dirname(w.name).split('/', 1)[1], 'S'))
                 for w in var_svd_f]
            VT = [tf.placeholder(shape=[None, None], dtype=tf.float32,
                                 name=os.path.join(os.path.dirname(w.name).split('/', 1)[1], 'VT'))
                  for w in var_svd_f]
            R = [tf.placeholder(dtype=tf.int32,
                                name=os.path.join(os.path.dirname(w.name).split('/', 1)[1], 'R'))
                 for w in var_svd_f]

        with tf.variable_scope('model_lowrank'):
            model_low, _ = build_model(FLAGS.model, input, nr_labels)
            var_svd_l, var_share_l = get_weights_for_svd(model_low.weights, svd_fc_last=FLAGS.svd_fc_last)
            mean_low, var_low, new_mean_low, new_var_low, assign_mean_low, assign_var_low, bn_inputs_low\
                = get_bn_variables(model_low)
            XX_T_low, Xl_low, X2l_low, M_low = compute_per_channel_stats(bn_inputs_low, format=FLAGS.format)
            build = build_low_rank_model(var_share_l, var_share_f, var_svd_l, U, S, VT, R, FLAGS.svd_decomposition)

        with tf.name_scope('evaluation'):
            logit = model_low(inputs=input)
            loss_sum = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=target, name='cross_entropy'),
                name='sum_cross_entropy')
            confusion_matrix = tf.matmul(target, tf.one_hot(tf.argmax(logit, 1), depth=nr_labels),
                                         transpose_a=True, name='confusion_matrix')

    # show model information
    print('var_svd_f = ', var_svd_f)
    print('var_svd_l = ', var_svd_l)
    print('bn_inputs_low = ', bn_inputs_low)
    print('mean_var_low = ', mean_low, var_low)

    # session
    with tf.Session(config=config) as sess:
        tf.train.Saver(var_list=model.weights).restore(sess, FLAGS.checkpoint)

        # svd
        start = time.time()
        W = sess.run(var_svd_f_mat)
        SVD = full_svd_cpu(W, algorithm='numpy_f_ccr')
        if SVD is None:
            print('SVD was failed and terminated.')
            sys.exit(1)
        print('svd time: %f sec' % (time.time() - start))
        rank_max = np.max([svd[1].shape[0] for svd in SVD])
        print('len(SVD) = ', len(SVD))
        print('rank_max = ', rank_max)

        # computing #params & macs for full network
        sort_func = select_sorting(FLAGS.sorting_criterion)
        params, macs, _, _ = compute_params_macs(model, svd_fc_last=FLAGS.svd_fc_last,
                                                 deco_mode=FLAGS.svd_decomposition, format=FLAGS.format, summary=True)
        print('params_macs = ', params, macs)

        # ordered singular values for ranking
        s = sort_func([svd[1] for svd in SVD])
        print('#total singular values = ', s.shape[0])

        # compute least error based ranking
        results = []
        for r in r_ratio:
            # compute rank
            rank = singular_value_rank(s, r, rank_min=FLAGS.rank_min)
            _, _, params_r, macs_r = compute_params_macs(model_low, rank=rank, svd_fc_last=FLAGS.svd_fc_last,
                                                         deco_mode=FLAGS.svd_decomposition, format=FLAGS.format)
            print('rank = ', rank)
            print('#params, #macs (original) = ', np.sum(params), np.sum(macs))
            print('#params, #macs (svd) = ', np.sum(params_r), np.sum(macs_r))
            print('#params (svd) = ', params_r)
            print('#macs (svd) = ', macs_r)

            # build low rank model
            feed_dict = {}
            for k, (svd, rnk) in enumerate(zip(SVD, rank)):
                feed_dict[U[k]], feed_dict[S[k]], feed_dict[VT[k]], feed_dict[R[k]] = svd[0], svd[1], svd[2], rnk
            sess.run(build, feed_dict=feed_dict)

            # computing population mean & covariance.
            start = time.time()
            myu, sg2 = compute_population_mean_var(sess, train_init_op, nr_batches_train, FLAGS.bn_batch_num_divider,
                                                   XX_T_low, Xl_low, X2l_low, M_low,
                                                   assign_mean_low, assign_var_low, new_mean_low, new_var_low)
            print('computing population mean & covariance time:', time.time() - start, 'sec')

            # evaluation
            start = time.time()
            res, confmat = compute_loss_acc(sess,
                                            valid_init_op,
                                            loss_sum,
                                            confusion_matrix,
                                            nr_samples_per_class_valid
                                            )
            print('{:<20}{:<20}{:<20}'.format('loss', 'acc_all', 'acc_class_ave'))
            print('{:<20.6}{:<20.6}{:<20.6}'.format(res[0], res[1], res[2]))
            print('evaluation time:', time.time() - start, 'sec')
            results.append([r, np.sum(params_r) / 1e+6, np.sum(macs_r) / 1e+6, 100 * res[2]])

        # save results
        results = ['r_ratio\tPARAMs(M)\tMACs(M)\tACC.(%)'] + \
                  ["\t".join(list(map("{:.6g}".format, res))) for res in results]
        file_path = FLAGS.out_file_path if FLAGS.out_file_path is not None \
            else 'res_%s_%s.txt' % (FLAGS.model, os.path.basename(FLAGS.checkpoint))
        with open(file_path, 'wt') as f:
            f.write('\n'.join(results))


if __name__ == '__main__':
    tf.app.run()
