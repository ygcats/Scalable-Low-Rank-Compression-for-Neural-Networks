import sys
import tensorflow as tf
import numpy as np
from model.vgg import vgg15
from common_np import sort_singular_values


def tf_config(gpuid):
    config = tf.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            visible_device_list=str(gpuid),  # specify GPU number
            allow_growth=True
        ),
        graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(
                global_jit_level=tf.OptimizerOptions.OFF  # or ON_1 in the future
                #opt_level=tf.OptimizerOptions.L1
            )
        )
    )
    tf.keras.backend.set_session(tf.Session(config=config))
    return config


def count_nr_variables(model, bn=False):
    vars = [s for s in model.weights
            if ('kernel:0' in s.name)
            or ('bias:0' in s.name)
            ]
    if bn:
        vars += [s for s in model.weights
                 if ('gamma:0' in s.name)
                 or ('beta:0' in s.name)
                 or ('moving_mean:0' in s.name)
                 or ('moving_variance:0' in s.name)
                ]
    return np.sum([np.product([xi.value for xi in x.get_shape()]) for x in vars]), vars


def compute_loss_acc(sess, init_op, loss, confmat, nr_samples_per_class):
    nr_labels = len(nr_samples_per_class)
    nr_samples = np.sum(nr_samples_per_class)
    l = 0.
    C = np.zeros(shape=[nr_labels, nr_labels], dtype='float32')

    sess.run(init_op)
    while True:
        try:
            res = sess.run([loss, confmat])
            l += res[0]
            C += res[1]
        except tf.errors.OutOfRangeError:
            break

    l /= nr_samples
    acc_all = C.trace() / nr_samples
    acc_matrix = C / np.reshape(nr_samples_per_class, (-1, 1))
    acc_class_ave = acc_matrix.trace() / nr_labels
    return [l, acc_all, acc_class_ave], C


def build_model(model_name, input, nr_labels, is_training=False):
    if model_name == 'vgg15':
        model = vgg15(nr_out_nodes=nr_labels, name=model_name)
    else:
        print('model: %s is not found' % model_name)
        sys.exit(1)
    output = model(inputs=input, training=is_training)  # first-call is needed to define weights
    return model, output


def tensor2mat(tensor, mode='KyKxCin_Cout'):
    if mode == 'KyKxCin_Cout':
        shape = tensor.get_shape().as_list()
        if len(shape) == 4:
            return tf.reshape(tensor, shape=[shape[0] * shape[1] * shape[2], shape[3]])
        elif len(shape) == 2:
            return tensor
        else:
            raise ValueError('invalid shape')

    elif mode == 'KxCin_KyCout':
        shape = tensor.get_shape().as_list()
        if len(shape) == 4:
            return tf.reshape(tf.transpose(tensor, perm=[1, 2, 0, 3]), shape=[shape[1] * shape[2], shape[0] * shape[3]])
        elif len(shape) == 2:
            return tensor
        else:
            raise ValueError('invalid shape')
    else:
        raise ValueError('unknown mode %r' % mode)


def mat2tensor(mat, shape, mode='KyKxCin_Cout'):
    if mode == 'KyKxCin_Cout':
        return tf.reshape(mat, shape=shape)

    elif mode == 'KxCin_KyCout':
        if len(shape) == 4:
            return tf.transpose(tf.reshape(mat, shape=[shape[1], shape[2], shape[0], shape[3]]), perm=[2, 0, 1, 3])
        elif len(shape) == 2:
            return tf.reshape(mat, shape=shape)
        else:
            raise ValueError('invalid shape')
    else:
        raise ValueError('unknown mode %r' % mode)


def build_low_rank_model(var_share_l, var_share_f, var_svd_l, U, S, VT, R, deco_mode):
    operations = [tf.assign(vsl, vsf) for vsl, vsf in zip(var_share_l, var_share_f)] + \
                 [tf.assign(w_l, mat2tensor(tf.matmul(u[:, :r], tf.matmul(tf.linalg.diag(s[:r]), vt[:r, :])),
                                            shape=w_l.get_shape().as_list(),
                                            mode=deco_mode)) for w_l, u, s, vt, r in zip(var_svd_l, U, S, VT, R)]
    with tf.control_dependencies(operations):
        return tf.no_op(name='build_low_rank_model')


def get_weights_for_svd(vars, share_bn_stats=False, svd_fc_last=False):
    svd = [s for s in vars if ('/kernel:0' in s.name) and ('fc_last' not in s.name)] if not svd_fc_last \
        else [s for s in vars if ('/kernel:0' in s.name)]
    shared = [s for s in vars if (s not in svd) and ('moving_' not in s.name)] if not share_bn_stats \
        else [s for s in vars if s not in svd]
    return svd, shared


def compute_params_macs(model, rank=None, svd_fc_last=False, deco_mode='Tucker2', format='NCHW', summary=False):
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

        # check args
        print('svd_fc_last: ', svd_fc_last)
        print('svd decomposition mode: ', deco_mode)
        print('data format: ', format)
        if summary:
            print(model.summary())
        if rank:
            conv_fc_layers = [l for l in layers if 'Conv2D' in l.__class__.__name__ or 'Dense' in l.__class__.__name__]
            print('#conv_fc_layers = ', len(conv_fc_layers))
            print('#rank = ', len(rank))
            if svd_fc_last and len(conv_fc_layers) != len(rank):
                raise ValueError('svd_fc_last and len(conv_fc_layers) != len(rank).')

        # compute params and macs
        N = 1   # batch size
        macs, params = [], []
        macs_r, params_r = [], []
        it = iter(rank) if rank is not None else None
        for l in layers:
            if 'Conv2D' in l.__class__.__name__:
                if format == 'NCHW':
                    _, _, Hin, Win = l.input_shape
                    _, _, Hout, Wout = l.output_shape
                else:
                    _, Hin, Win, _ = l.input_shape
                    _, Hout, Wout, _ = l.output_shape
                Ky, Kx, Cin, Cout = l.weights[0].get_shape().as_list()
                params.append(Ky * Kx * Cin * Cout)
                macs.append(Ky * Kx * Cin * Cout * Hout * Wout * N)
                # low-rank
                if rank is not None:
                    r = next(it)
                    r = r[0] if type(r) == list and len(r) == 1 else r
                    if deco_mode == 'KyKxCin_Cout':
                        params_r.append((Ky * Kx * Cin + Cout) * r)
                        macs_r.append(Ky * Kx * Cin * r * Hout * Wout * N + 1 * 1 * r * Cout * Hout * Wout * N)
                    elif deco_mode == 'KxCin_KyCout':
                        params_r.append((Kx * Cin + Ky * Cout) * r)
                        macs_r.append(1 * Kx * Cin * r * Hin * Wout * N + Ky * 1 * r * Cout * Hout * Wout * N if Ky > 1
                                      else 1 * Kx * Cin * r * Hout * Wout * N + Ky * 1 * r * Cout * Hout * Wout * N)
                    elif deco_mode == 'Tucker2':
                        if type(r) != list:
                            params_r.append(Ky * Kx * Cin * r + r * Cout)
                            macs_r.append(Ky * Kx * Cin * r * Hout * Wout * N
                                          + 1 * 1 * r * Cout * Hout * Wout * N)
                        elif type(r) == list and len(r) == 2:
                            params_r.append(Cin * r[0] + Ky * Kx * r[0] * r[1] + r[1] * Cout)
                            macs_r.append(1 * 1 * Cin * r[0] * Hin * Win * N
                                          + Ky * Kx * r[0] * r[1] * Hout * Wout * N
                                          + 1 * 1 * r[1] * Cout * Hout * Wout * N)
                        else:
                            raise ValueError('invalid ranks')
                    else:
                        raise ValueError('invalid svd decomposition mode: %s' % deco_mode)
            elif 'Dense' in l.__class__.__name__:
                ks = l.weights[0].get_shape().as_list()
                params.append(ks[0] * ks[1])
                macs.append(ks[0] * ks[1] * N)
                # low-rank
                if rank is not None:
                    if svd_fc_last or 'fc_last' not in l.weights[0].name:
                        m, n = ks[0], ks[1]
                        r = next(it)
                        r = r[0] if type(r) == list and len(r) == 1 else r
                        params_r.append((m + n) * r)
                        macs_r.append((m + n) * r * N)
                    else:
                        params_r.append(ks[0] * ks[1])
                        macs_r.append(ks[0] * ks[1] * N)
    except ValueError as err:
        print(err)
        return [], [], [], []
    return params, macs, params_r, macs_r


def select_sorting(sorting_criterion):
    sort_func = None

    if sorting_criterion == 'sv':
        print('sorting wih singular values')
        sort_func = sort_singular_values
    else:
        print('invalid criterion: ', sorting_criterion)

    return sort_func


