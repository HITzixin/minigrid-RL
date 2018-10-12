#!/usr/bin/env python3

import os
import gym
import operator
from gym import spaces
from baselines import logger
from baselines.bench import Monitor
from baselines.common.cmd_util import arg_parser
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from algo.a2c_ent import learn
from policies import CnnPolicy_grid, CnnPolicy_pred
try:
    import gym_minigrid
    from gym_minigrid.wrappers import *
except:
    pass


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + 100 * rank)
        # print('make_env')
        # print(env.observation_space)
        # Maxime: until RL code supports dict observations, squash observations into a flat vector
        # if isinstance(env.observation_space, spaces.Dict):
        #    env = FlatObsWrapper(env)
        # print(env.observation_space)
        env = Monitor(env, logger.get_dir() and os.path.join(
            logger.get_dir(), str(rank)))
        return env
    return _thunk


def train(env_id, save_name, num_timesteps, seed, policy, lrschedule, sil_update, sil_beta, num_env):
    policy_fn = CnnPolicy_grid
    # env_args = {'episode_life': False, 'clip_rewards': False}
    env = gym.make(env_id)
    obs = env.reset()
    ob_space = obs["image"].shape
    ac_space = env.action_space
    env.close()
    # print(env.observation_space)
    print('ob_space:', ob_space)
    print('num act:', ac_space)
    envs = [make_env(env_id, seed, i) for i in range(num_env)]
    if num_env > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)
    # obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])
    # print('obs_shape_stack:',obs_shape)
    learn(policy_fn, envs, seed, ob_space, ac_space, save_name=save_name, nsteps=5,
          total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule, lr=7e-4)

    envs.close()


def main():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID',
                        default='MiniGrid-MultiRoom-N2-S4-v0')  # MiniGrid-MultiRoom-N4-v0
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=int(30e6))
    parser.add_argument('--policy', help='Policy architecture',
                        choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule',
                        choices=['constant', 'linear'], default='constant')
    parser.add_argument('--sil-update', type=int, default=4,
                        help="Number of updates per iteration")
    parser.add_argument('--sil-beta', type=float,
                        default=0.1, help="Beta for weighted IS")
    parser.add_argument('--log', default='./log')
    parser.add_argument('--save_name', default='MultiRoomN2S4_a2c',
                        help="Path for saved model")

    args = parser.parse_args()
    logger.configure(dir=args.log)
    train(args.env, args.save_name, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lrschedule=args.lrschedule,
          sil_update=args.sil_update, sil_beta=args.sil_beta,
          num_env=16)


if __name__ == '__main__':
    main()
