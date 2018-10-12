import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
#from baselines.common.input import observation_input
from utils import cat_entropy

import tensorflow as tf
from gym.spaces import Discrete, Box


def observation_input(ob_space, batch_size=None, name='Ob'):
    '''
    Build observation input with encoding depending on the
    observation space type
    Params:

    ob_space: observation space (should be one of gym.spaces)
    batch_size: batch size for input (default is None, so that resulting input placeholder can take tensors with any batch size)
    name: tensorflow variable name for input placeholder

    returns: tuple (input_placeholder, processed_input_tensor)
    '''
    input_shape = (batch_size,) + ob_space
    input_x = tf.placeholder(shape=input_shape, dtype=tf.int32, name=name)
    processed_x = tf.to_float(input_x)
    print('Box')
    print(input_shape)
    print(input_x)
    print(processed_x.shape)
    return input_x, processed_x


def nature_cnn(unscaled_images, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2,
                    init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1,
                    init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        X, processed_x = observation_input(ob_space, nbatch)
        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        self.pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value


class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        M = tf.placeholder(tf.float32, [nbatch])  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2])  # states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            vf = fc(h5, 'v', 1)
            self.pd, self.pi = self.pdtype.pdfromlatent(h5)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm * 2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X: ob, S: state, M: mask})

        def value(ob, state, mask):
            return sess.run(v0, {X: ob, S: state, M: mask})

        self.X = X
        self.M = M
        self.S = S
        self.vf = vf
        self.step = step
        self.value = value


class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        #X, processed_x = observation_input(ob_space, nbatch)
        X, processed_x = observation_input(ob_space, None)
        print('X:', X.shape)
        print('processed_X:', processed_x.shape)
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(processed_x, **conv_kwargs)
            vf = fc(h, 'v', 1)[:, 0]
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        self.entropy = cat_entropy(self.pi)

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        def neg_log_prob(actions):
            return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pi, labels=actions)

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.neg_log_prob = neg_log_prob


class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, None)
        #X = tf.placeholder(shape=input_shape,dtype=tf.float32, name='ob')
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            processed_x = tf.layers.flatten(processed_x)
            pi_h1 = activ(fc(processed_x, 'pi_fc1',
                             nh=64, init_scale=np.sqrt(2)))
            pi_h2 = activ(fc(pi_h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            vf_h1 = activ(fc(processed_x, 'vf_fc1',
                             nh=64, init_scale=np.sqrt(2)))
            vf_h2 = activ(fc(vf_h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(vf_h2, 'vf', 1)[:, 0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_h2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None
        self.entropy = cat_entropy(self.pi)

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        def neg_log_prob(actions):
            return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pi, labels=actions)

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.neg_log_prob = neg_log_prob


def cnn_grid(unscaled_images, **conv_kwargs):
    """
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=2),
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
    nn.ReLU(),
    nn.Conv2d(in_channels=32, out_channels=self.image_embedding_size, kernel_size=(2, 2)),
    nn.ReLU()
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=16, rf=2, stride=1, init_scale=np.sqrt(2),
                   **conv_kwargs))
    h1 = tf.nn.max_pool(h, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='VALID', name='m1')
    h2 = activ(conv(h1, 'c2', nf=32, rf=2, stride=1,
                    init_scale=np.sqrt(2), **conv_kwargs))
    h3 = activ(conv(h2, 'c3', nf=64, rf=2, stride=1,
                    init_scale=np.sqrt(2), **conv_kwargs))
    h3 = conv_to_fc(h3)
    return h3
    # return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))


class CnnPolicy_grid(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        #X, processed_x = observation_input(ob_space, nbatch)
        X, processed_x = observation_input(ob_space, None)
        print('X:', X.shape)
        print('processed_X:', processed_x.shape)
        with tf.variable_scope("model", reuse=reuse):
            h = cnn_grid(processed_x, **conv_kwargs)
            actor_l1 = fc(h, 'actor', nh=64, init_scale=np.sqrt(2))
            actor_l2 = tf.nn.tanh(actor_l1)
            #actor_l3 = fc(actor_l2, 'actor2', nh=action_space.n, init_scale=np.sqrt(2))
            critic_l1 = fc(h, 'critic', nh=64, init_scale=np.sqrt(2))
            critic_l2 = tf.nn.tanh(critic_l1)
            critic_l3 = fc(critic_l2, 'critic2', nh=1, init_scale=np.sqrt(2))
            vf = critic_l3[:, 0]
            self.pd, self.pi = self.pdtype.pdfromlatent(
                actor_l2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        self.entropy = cat_entropy(self.pi)

        def step(ob, *_args, **_kwargs):
            a, v, entropy, neglogp = sess.run(
                [a0, vf, self.entropy, neglogp0], {X: ob})
            return a, v, entropy, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        def neg_log_prob(actions):
            return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pi, labels=actions)

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.neg_log_prob = neg_log_prob


class CnnPolicy_pred(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, **conv_kwargs):  # pylint: disable=W0613
        self.pdtype = make_pdtype(ac_space)
        #X, processed_x = observation_input(ob_space, nbatch)
        X, processed_x = observation_input(ob_space, None)
        print('X:', X.shape)
        print('processed_X:', processed_x.shape)
        print('ac_space:', ac_space.n)
        with tf.variable_scope("model", reuse=reuse):
            h = cnn_grid(processed_x, **conv_kwargs)
            actor_l1 = fc(h, 'actor', nh=64, init_scale=np.sqrt(2))
            self.phi = actor_l2 = tf.nn.tanh(actor_l1)
            #actor_l3 = fc(actor_l2, 'actor2', nh=action_space.n, init_scale=np.sqrt(2))
            critic_l1 = fc(h, 'critic', nh=64, init_scale=np.sqrt(2))
            critic_l2 = tf.nn.tanh(critic_l1)
            critic_l3 = fc(critic_l2, 'critic2', nh=1, init_scale=np.sqrt(2))
            vf0 = critic_l3
            vf = vf0[:, 0]
            self.pd, self.pi = self.pdtype.pdfromlatent(
                actor_l2, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        self.entropy = cat_entropy(self.pi)

        with tf.variable_scope("planning", reuse=reuse):
            # predict next action
            a0_onehot = tf.stop_gradient(tf.one_hot(a0, ac_space.n, axis=-1))
            f = tf.concat([self.phi, a0_onehot], axis=1)
            self.pd_p, self.pi_p = self.pdtype.pdfromlatent(
                f, init_scale=0.01)
            self.ap = self.pd_p.sample()

        def step(ob, *_args, **_kwargs):
            a, v, neglogp, ap = sess.run([a0, vf, neglogp0, self.ap], {X: ob})
            return a, v, ap, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        def neg_log_prob(actions):
            return tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pi, labels=actions)

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
        self.neg_log_prob = neg_log_prob
