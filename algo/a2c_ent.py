import os.path as osp
import time
import joblib
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from abc import ABC, abstractmethod

from utils import discount_with_dones
from utils import Scheduler, make_path, find_trainable_variables
from utils import cat_entropy, mse, EpisodeStats


class AbstractEnvRunner(ABC):
    def __init__(self, *, env, model, nsteps, ob_space):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.batch_ob_shape = (nenv * nsteps,) + ob_space
        self.obs = np.zeros((nenv,) + ob_space, dtype=np.uint8)
        obs = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
                 alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear',
                 summary_dir=None):

        sess = tf_util.make_session()
        nbatch = nenvs * nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, reuse=False)
        train_model = policy(sess, ob_space, ac_space,
                             nenvs * nsteps, nsteps, reuse=True)

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=train_model.pi, labels=A)
        pg_loss = tf.reduce_mean(ADV * neglogpac)
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        entropy = tf.reduce_mean(cat_entropy(train_model.pi))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(
            learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        # storing summaries
        episode_reward = tf.placeholder("float")
        tf.summary.scalar("policy_loss", pg_loss)
        tf.summary.scalar("entropy", entropy)
        tf.summary.scalar("value_loss", vf_loss)
        tf.summary.scalar("episode_reward", episode_reward)
        summary_op = tf.summary.merge_all()

        def train(obs, states, mean_reward, rewards, masks, actions, values):
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X: obs, A: actions,
                      ADV: advs, R: rewards, LR: cur_lr,
                      episode_reward: mean_reward}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, summary, _ = sess.run(
                [pg_loss, vf_loss, entropy, summary_op, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy, summary

        def save(save_path):
            ps = sess.run(params)
            make_path(osp.dirname(save_path))
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)
        self.train_writer = tf.summary.FileWriter(summary_dir, sess.graph)


class Runner(AbstractEnvRunner):
    """
        the gym-minigrid envs: the reward  [0.1,1], choose entropy coef=1e-3
    """

    def __init__(self, env, model, ob_space, nsteps=20, gamma=0.99, entropy_coef=1e-4):
        super().__init__(env=env, model=model, nsteps=nsteps, ob_space=ob_space)
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_raw_rewards = [], [], [], [], [], []
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, entropy, states, _ = self.model.step(
                self.obs, self.states, self.dones)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs_all, raw_rewards, dones, _ = self.env.step(actions)
            obs = [obs_index['image'] for obs_index in obs_all]
            obs = np.asarray(obs)
            #rewards = raw_rewards + entropy * self.entropy_coef
            rewards = raw_rewards
            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n] * 0
            self.obs = obs
            mb_rewards.append(rewards)
            mb_raw_rewards.append(raw_rewards)
        mb_dones.append(self.dones)
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(
            1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_raw_rewards = np.asarray(
            mb_raw_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(
            self.obs, self.states, self.dones).tolist()
        # discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(
                    rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_raw_rewards = mb_raw_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_raw_rewards


def learn(policy, env, seed, ob_space, ac_space, save_name, nsteps=5,
          total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
          lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99,
          log_interval=100):
    set_global_seeds(seed)

    nenvs = env.num_envs
    #ob_space = env.observation_space
    #ac_space = env.action_space
    save_dir = './model/' + save_name + '.ckt'
    summary_dir = './summary/' + save_name

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                  max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule,
                  summary_dir=summary_dir)
    runner = Runner(env, model, ob_space=ob_space, nsteps=nsteps, gamma=gamma)

    nbatch = nenvs * nsteps
    tstart = time.time()
    train_writer = model.train_writer

    episode_stats = EpisodeStats(nsteps, nenvs)
    for update in range(1, total_timesteps // nbatch + 1):
        obs, states, rewards, masks, actions, values, raw_rewards = runner.run()
        episode_stats.feed(raw_rewards, masks)
        mean_reward = episode_stats.mean_reward()
        mean_reward = np.asarray(mean_reward, dtype=np.float32)

        policy_loss, value_loss, policy_entropy, summary = model.train(
            obs, states, mean_reward, rewards, masks, actions, values)
        train_writer.add_summary(summary, update * nbatch)

        nseconds = time.time() - tstart
        fps = int((update * nbatch) / nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update * nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.record_tabular(
                "episode_reward", episode_stats.mean_reward())
            logger.record_tabular(
                "episode_length", episode_stats.mean_length())
            logger.dump_tabular()
            model.save(save_dir)
    env.close()
    return model
