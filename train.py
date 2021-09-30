import os
import sys
import time
import random as rn

import tensorflow as tf

from dataset.img_reader import read_path_label, build_input_pipeline_eval, build_input_pipeline_train, parse_func_cifar
from common_np import full_svd_cpu, singular_value_rank
from common_tf import tf_config, count_nr_variables, compute_loss_acc, build_model, \
    tensor2mat, mat2tensor, build_low_rank_model, get_weights_for_svd, select_sorting


# cifar10
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('logdir', None, 'output directory for log files')
tf.flags.DEFINE_integer('device', 0, 'gpu_id')
tf.flags.DEFINE_string('train_path', None, 'path to a txt file for training')
tf.flags.DEFINE_string('valid_path', None, 'path to a txt file for validation')
tf.flags.DEFINE_integer('cpu_threads', 8, '# of cpu threads')
tf.flags.DEFINE_integer('cpu_threads_np', 4, '# of cpu threads')
tf.flags.DEFINE_float('momentum', 0.9, 'μ in momentum SGD')
tf.flags.DEFINE_bool('use_nesterov', True, 'nesterov acceleration in momentum SGD')
tf.flags.DEFINE_float('l2_lambda', 0.0005, 'l2-regularization')
tf.flags.DEFINE_integer('random_seed', 0, 'random seed')
tf.flags.DEFINE_integer('eval_inter_epochs_train', 1, 'evaluation interval epochs for training data')
tf.flags.DEFINE_integer('eval_inter_epochs_valid', 1, 'evaluation interval epochs for validation data')
tf.flags.DEFINE_integer('checkpoint_max_keep', 1, 'the number of maximum checkpoints to keep')
tf.flags.DEFINE_bool('drop_remainder', False, 'drop the last mini-batch')

tf.flags.DEFINE_string('model', 'vgg15', 'network architecture')
tf.flags.DEFINE_integer('epochs', 200, 'the number of epochs')
tf.flags.DEFINE_integer('batch_size', 128, 'size of mini-batch')
tf.flags.DEFINE_integer('batch_size_eval', 1024, 'size of mini-batch for evaluation step')
tf.flags.DEFINE_float('init_lr', 0.1, 'initial learning rate')
tf.flags.DEFINE_string('lr_scheduler', 'multistep', 'learning rate scheduler')
tf.flags.DEFINE_list('multistep_lr_decay_epochs', [60, 120, 160], 'list of epochs for multistep learning rate decay')
tf.flags.DEFINE_list('multistep_lr_decay_rate', [0.2, 0.2, 0.2], 'list of decay rate for multistep learning rate decay')
tf.flags.DEFINE_integer('step_lr_decay_epochs', 30, 'step learning rate decay is applied every this number of epochs')
tf.flags.DEFINE_float('step_lr_decay_rate', 0.1, 'step learning rate decay is applied with this rate')
tf.flags.DEFINE_float('polynomial_lr_rate_end', 0, 'end value of learning rate for polynomial scheduler')
tf.flags.DEFINE_float('polynomial_lr_power', 1, 'power for polynomial scheduler')

tf.flags.DEFINE_float('r_lower', 0.01, 'lower bound of rank ratio')
tf.flags.DEFINE_float('r_upper', 0.25, 'upper bound of rank ratio')
tf.flags.DEFINE_integer('rank_min', 5, 'minimum rank for svd')
tf.flags.DEFINE_bool('svd_fc_last', True, 'apply svd to a matrix in the last fc layer')
tf.flags.DEFINE_float('lowrank_loss', 0.5, 'strength of low-rank loss (0-1)')
tf.flags.DEFINE_string('sorting_criterion', 'sv', 'mode of sorting bases')
tf.flags.DEFINE_string('svd_decomposition', 'KyKxCin_Cout', 'mode of svd decomposition')
tf.flags.DEFINE_float('svd_bp_eps', 0.99, 'a constant for numerical stability of SVD-backprop.')

os.environ["OMP_NUM_THREADS"] = str(FLAGS.cpu_threads_np)         # export OMP_NUM_THREADS=FLAGS.cpu_threads
os.environ["OPENBLAS_NUM_THREADS"] = str(FLAGS.cpu_threads_np)    # export OPENBLAS_NUM_THREADS=FLAGS.cpu_threads
os.environ["MKL_NUM_THREADS"] = str(FLAGS.cpu_threads_np)         # export MKL_NUM_THREADS=FLAGS.cpu_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = str(FLAGS.cpu_threads_np)  # export VECLIB_MAXIMUM_THREADS=FLAGS.cpu_threads
os.environ["NUMEXPR_NUM_THREADS"] = str(FLAGS.cpu_threads_np)     # export NUMEXPR_NUM_THREADS=FLAGS.cpu_threads
import numpy as np
np.__config__.show()


def print_evaluation_results(epoch, loss_acc_train, loss_acc_valid, learning_rate):
    if epoch == 1:
        print('{:<10}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}{:<20}'.format(
            'epoch', 'loss_train', 'acc_all_train', 'acc_class_train',
            'loss_valid', 'acc_all_valid', 'acc_class_valid', 'learning_rate (next)'))
    res = loss_acc_train if loss_acc_train is not None else ['None', 'None', 'None']
    res += loss_acc_valid if loss_acc_valid is not None else ['None', 'None', 'None']
    print('{:<10}{:<20.6}{:<20.6}{:<20.6}{:<20.6}{:<20.6}{:<20.6}{:<20.6}'.format(
        epoch, res[0], res[1], res[2], res[3], res[4], res[5], learning_rate))


def tf_learning_rate(lr_scheduler, global_step, nr_iterations_per_epoch, init_lr,
                     step_lr_decay_epochs=None, step_lr_decay_rate=None,
                     multistep_lr_decay_rate=None, multistep_lr_decay_epochs=None,
                     epochs=None, polynomial_lr_rate_end=None, polynomial_lr_power=None
                     ):
    if lr_scheduler == 'step':
        learning_rate = tf.train.exponential_decay(init_lr, global_step,
                                                   step_lr_decay_epochs * nr_iterations_per_epoch,
                                                   step_lr_decay_rate, staircase=True,
                                                   name='step_lr_decay')
    elif lr_scheduler == 'multistep':
        steps, rates = [], []
        l = len(multistep_lr_decay_rate)
        if len(multistep_lr_decay_epochs) != l:
            print('multistep_lr_decay_epochs and multistep_lr_decay_rate must have same length.')
            sys.exit(1)
        steps = [int(v) * nr_iterations_per_epoch for v in multistep_lr_decay_epochs]
        rates = [init_lr * np.prod(list(map(float, multistep_lr_decay_rate))[:k]) for k in range(l + 1)]
        learning_rate = tf.case(
            [(tf.less(global_step, steps[k]), lambda j=k: tf.constant(rates[j])) for k in range(len(steps))],
            default=lambda: tf.constant(rates[-1]),
            name='multistep_lr_decay')
    elif lr_scheduler == 'polynomial':
        learning_rate = tf.train.polynomial_decay(init_lr, global_step,
                                                  epochs * nr_iterations_per_epoch,
                                                  end_learning_rate=polynomial_lr_rate_end,
                                                  power=polynomial_lr_power, cycle=False,
                                                  name='polynomial_lr_decay')
    else:
        print('lr_scheduler: %s is not found' % lr_scheduler)
        sys.exit(1)
    return learning_rate


# dWr: 2D (m x n) tensor for the gradient of a low rank weight matrix
# U:    2D (m x min(m, n)) tensor for left singular vectors
# S:    1D (min(m, n)) tensor for singular values
# VT:   2D (min(m, n) x n) tensor for right singular vectors
# r:    scalar tensor for the rank of a weight matrix
# eps:  a small constant for numerical stability
def grad_trancated_SVD_tf(dWr, U, S, VT, r, eps):

    def portrait(dWr, U, S, VT, r):
        n = tf.shape(dWr)[1]
        S2 = tf.reshape(tf.square(S), [-1, 1])
        x2 = tf.transpose(S2[r:n, :]) / tf.maximum(S2[:r, :], 1.0e-15)
        x2 = tf.minimum(x2, eps)  # clipping
        SrSrK = 1. / (1. - x2)    # clipping
        SrKSrn = tf.sqrt(x2) * SrSrK
        KSrnSrn = x2 * SrSrK

        VrnT = VT[r:n, :]
        Ur = U[:, :r]
        VrT = VT[:r, :]
        Urn = U[:, r:n]

        A = tf.matmul(tf.matmul(VrT, dWr, transpose_b=True), Urn)
        B = tf.matmul(tf.matmul(Ur, dWr, transpose_a=True), VrnT, transpose_b=True)

        return tf.matmul(dWr, tf.matmul(VrT, VrT, transpose_a=True)) + \
               tf.matmul(tf.matmul(Ur, SrKSrn * A + SrSrK * B), VrnT) + \
               tf.matmul(tf.matmul(Urn, KSrnSrn * A + SrKSrn * B, transpose_b=True), VrT)

    return tf.cond(tf.greater(tf.shape(dWr)[1], tf.shape(dWr)[0]),                                              # m < n?
                   lambda: tf.transpose(portrait(tf.transpose(dWr), tf.transpose(VT), S, tf.transpose(U), r)),  # m < n
                   lambda: portrait(dWr, U, S, VT, r)                                                           # m >= n
                   )


def backprop_truncated_SVD(grads_low, U, S, VT, R, var_svd_l, deco_mode='KyKxCin_Cout', eps=1.0):
    grads_low_bp = []
    u_iter = iter(U)
    s_iter = iter(S)
    vt_iter = iter(VT)
    r_iter = iter(R)

    for g in grads_low:
        dw, w = g
        if w in var_svd_l:
            u = next(u_iter)
            s = next(s_iter)
            vt = next(vt_iter)
            r = next(r_iter)
            dw_bp = mat2tensor(mat=grad_trancated_SVD_tf(tensor2mat(tensor=dw, mode=deco_mode), u, s, vt, r, eps),
                               shape=dw.get_shape().as_list(),
                               mode=deco_mode)
            grads_low_bp.append((dw_bp, w))
        else:
            grads_low_bp.append(g)
    return grads_low_bp


def main(argv):
    # parameters
    np.random.seed(FLAGS.random_seed)
    rn.seed(FLAGS.random_seed)
    log_dir = FLAGS.logdir
    config = tf_config(FLAGS.device)

    # load dataset
    train_paths, train_labels = read_path_label(FLAGS.train_path)
    valid_paths, valid_labels = read_path_label(FLAGS.valid_path)
    nr_labels = len(np.unique(train_labels))
    nr_samples_per_class_train = np.eye(nr_labels, dtype='float32')[train_labels].sum(axis=0)
    nr_samples_per_class_valid = np.eye(nr_labels, dtype='float32')[valid_labels].sum(axis=0)
    nr_iterations_per_epoch = np.floor(len(train_paths) / FLAGS.batch_size).astype('int32') if FLAGS.drop_remainder \
        else np.ceil(len(train_paths) / FLAGS.batch_size).astype('int32')
    print('# of labels = ', nr_labels)
    print('# of training samples = ', len(train_paths))
    print('# of validation samples = ', len(valid_paths))
    print('# of iterations per epoch = ', nr_iterations_per_epoch)
    print('# of training samples per class = ', nr_samples_per_class_train)
    print('# of validation samples per class = ', nr_samples_per_class_valid)
    print('model is "%s"' % FLAGS.model)

    #make graph
    with tf.device('/cpu:0'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        with tf.variable_scope('data'):
            # dataset
            parser_train, parser_eval = parse_func_cifar(nr_labels)
            train_set = build_input_pipeline_train(
                parser_train, train_paths, train_labels, FLAGS.batch_size, FLAGS.cpu_threads,
                drop_remainder=FLAGS.drop_remainder)
            train_set_eval = build_input_pipeline_eval(
                parser_eval, train_paths, train_labels, FLAGS.batch_size_eval, FLAGS.cpu_threads)
            valid_set = build_input_pipeline_eval(
                parser_eval, valid_paths, valid_labels, FLAGS.batch_size_eval, FLAGS.cpu_threads)

            # make an iterator by data structure (this can be used for both training and validation datasets)
            iterator = tf.data.Iterator.from_structure(
                train_set.output_types,
                ([None] + train_set.output_shapes[0].as_list()[1:],
                 [None] + train_set.output_shapes[1].as_list()[1:])
            )

            # make initializers for training and validation datasets
            train_init_op = iterator.make_initializer(train_set)
            train_init_op_eval = iterator.make_initializer(train_set_eval)
            valid_init_op = iterator.make_initializer(valid_set)

            # minibatch data
            input, target = iterator.get_next()

    with tf.device('/gpu:0'):
        tf.set_random_seed(FLAGS.random_seed)
        with tf.variable_scope('model'):
            model, logit_org = build_model(FLAGS.model, input, nr_labels, is_training=True)
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
            build = build_low_rank_model(var_share_l, var_share_f, var_svd_l, U, S, VT, R, FLAGS.svd_decomposition)

        with tf.name_scope('loss'):
            # full-rank loss
            loss_org = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_org, labels=target, name='cross_entropy_org'),
                name='mean_cross_entropy_org')
            regularization = tf.reduce_sum(model.losses, name='regularization')
            with tf.control_dependencies([build]):
                # low-rank loss
                logit_low = model_low(inputs=input, training=True)
                loss_low = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_low, labels=target, name='cross_entropy_low'),
                    name='mean_cross_entropy_low')

        with tf.name_scope('update'):
            # learning rate, optimizer, and BN stats. updater
            learning_rate = tf_learning_rate(FLAGS.lr_scheduler, global_step, nr_iterations_per_epoch, FLAGS.init_lr,
                                             step_lr_decay_epochs=FLAGS.step_lr_decay_epochs,
                                             step_lr_decay_rate=FLAGS.step_lr_decay_rate,
                                             multistep_lr_decay_rate=FLAGS.multistep_lr_decay_rate,
                                             multistep_lr_decay_epochs=FLAGS.multistep_lr_decay_epochs,
                                             epochs=FLAGS.epochs,
                                             polynomial_lr_rate_end=FLAGS.polynomial_lr_rate_end,
                                             polynomial_lr_power=FLAGS.polynomial_lr_power)
            opt = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=FLAGS.momentum, use_nesterov=FLAGS.use_nesterov)

            # full-rank gradient
            grads = opt.compute_gradients(loss_org,
                                          [s for s in model.weights if ('moving_' not in s.name)])
            # low-rank gradient
            grads_low = opt.compute_gradients(loss_low,
                                              [s for s in model_low.weights if ('moving_' not in s.name)])
            grads_low = backprop_truncated_SVD(grads_low, U, S, VT, R, var_svd_l,
                                               deco_mode=FLAGS.svd_decomposition, eps=FLAGS.svd_bp_eps)
            grads_low = [(tf.linalg.norm(g[0]) / tf.maximum(tf.linalg.norm(gl[0]), 1.0e-15) * gl[0], gl[1])
                         if gl[1] in var_svd_l
                         else gl for g, gl in zip(grads, grads_low)]  # normalize low-rank gradient

            # regularization gradient
            grads_reg = opt.compute_gradients(regularization,
                                              [s for s in model.weights if ('moving_' not in s.name)])

            # aggregate gradients
            alpha = tf.constant(FLAGS.lowrank_loss, name='weight_param_of_low_rank_loss')
            grads = [((1. - alpha) * x[0] + alpha * y[0] + FLAGS.l2_lambda * z[0], x[1]) if z[0] is not None
                     else ((1. - alpha) * x[0] + alpha * y[0], x[1])
                     for x, y, z in zip(grads, grads_low, grads_reg)]
            train_step = tf.group([opt.apply_gradients(grads, global_step=global_step), model.updates])
            train_step_base = tf.group([opt.minimize(loss_org + FLAGS.l2_lambda * regularization,
                                                     global_step=global_step), model.updates])

        with tf.name_scope('evaluation'):
            logit_eval = model(inputs=input, training=False)
            loss_sum = tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logit_eval, labels=target, name='cross_entropy_eval'),
                name='sum_cross_entropy_eval')
            confusion_matrix = tf.matmul(target, tf.one_hot(tf.argmax(logit_eval, 1), depth=nr_labels),
                                         transpose_a=True, name='confusion_matrix')

    # show model information
    params, _ = count_nr_variables(model, bn=False)
    print('#params = ', params)
    print('#svd layers = ', len(var_svd_f) if len(var_svd_f) == len(var_svd_l) else -1)
    print(tf.global_variables())
    print(tf.trainable_variables())
    print(model.weights)
    print(model.losses)
    print(model.updates)

    # sorting_criterion
    sort_func = select_sorting(FLAGS.sorting_criterion)

    #session
    checkpoint_dir = log_dir
    start = time.time()
    with tf.train.MonitoredTrainingSession(config=config,
                                           checkpoint_dir=checkpoint_dir,
                                           scaffold=tf.train.Scaffold(
                                               saver=tf.train.Saver(max_to_keep=FLAGS.checkpoint_max_keep)),
                                           save_summaries_steps=nr_iterations_per_epoch,
                                           save_checkpoint_steps=nr_iterations_per_epoch
                                           ) as sess:
        epoch_start = sess.run(global_step) // nr_iterations_per_epoch
        for i in range(epoch_start, FLAGS.epochs):
            iter_start = sess.run(global_step) % nr_iterations_per_epoch
            sess.run(train_init_op)  # initialize iterator by train_set (switch to train_set)
            for j in range(iter_start, nr_iterations_per_epoch):
                if FLAGS.r_lower > 0 and FLAGS.r_upper > 0:
                    W = sess.run(var_svd_f_mat)
                    SVD = full_svd_cpu(W, algorithm='numpy_f_ccr')
                    if SVD is not None:
                        s = sort_func([svd[1] for svd in SVD])
                        ratio = np.random.uniform(low=FLAGS.r_lower, high=FLAGS.r_upper)
                        rank = singular_value_rank(s, ratio, rank_min=FLAGS.rank_min)
                        feed_dict = {}
                        for k, (svd, r) in enumerate(zip(SVD, rank)):
                            feed_dict[U[k]], feed_dict[S[k]], feed_dict[VT[k]], feed_dict[R[k]] = \
                                svd[0], svd[1], svd[2], r
                        sess.run(train_step, feed_dict=feed_dict)
                    else:
                        print('switching to regular training step for global_step %d.' % sess.run(global_step))
                        sess.run(train_step_base)
                else:
                    sess.run(train_step_base)

            # evaluation on training & validation set
            loss_acc_train, _ = compute_loss_acc(sess, train_init_op_eval, loss_sum, confusion_matrix,
                                                 nr_samples_per_class_train) \
                if i + 1 == 1 or i + 1 == FLAGS.epochs or (i + 1) % FLAGS.eval_inter_epochs_train == 0 else [None, None]
            loss_acc_valid, _ = compute_loss_acc(sess, valid_init_op, loss_sum, confusion_matrix,
                                                 nr_samples_per_class_valid) \
                if i + 1 == 1 or i + 1 == FLAGS.epochs or (i + 1) % FLAGS.eval_inter_epochs_valid == 0 else [None, None]

            # log
            print_evaluation_results(i + 1, loss_acc_train, loss_acc_valid, sess.run(learning_rate))
    print('training time:', time.time() - start, 'sec')


if __name__ == '__main__':
    tf.app.run()
