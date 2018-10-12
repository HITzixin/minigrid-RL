#!/usr/bin/env python3

import argparse
import gym
from gym import spaces
import time
import random
#from baselines.minigrid.a2c.a2c_sil import Model
from baselines.minigrid.a2c.a2c import Model
from baselines.minigrid.a2c.policies import CnnPolicy_grid
try:
    import gym_minigrid
    from gym_minigrid.wrappers import *
except:
    pass

import utils

# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", default='MiniGrid-DoorKey-5x5-v0',
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", default='./model/minigrid/minigrid5x5_a2c.ckt',
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
args = parser.parse_args()

# Generate environment

env = gym.make(args.env)
env.seed(args.seed)
obs_all = env.reset()
obs = obs_all['image']
ob_space = obs.shape
ac_space = env.action_space
num_act = env.action_space.n
print('ob_space:', ob_space)
print('ac_space:', ac_space)
print('num_act:', num_act)
# Define agent
agent = Model(policy=CnnPolicy_grid, ob_space=ob_space,
              ac_space=ac_space, nenvs=1, nsteps=1)
agent.load(args.model)
print('Load pretrained model successfully')
# Run the agent

done = True
action = 0
state = np.zeros((1, 7, 7, 3))
while True:
    if done:
        obs_all = env.reset()
        obs = obs_all['image']
        num_step = 0
        #print("Instr:", obs["mission"])
    state[0, :, :, :] = obs
    renderer = env.render("human")
    action, _, _, _ = agent.step(state)
    #action = random.randint(0, num_act - 1)
    obs_all, reward, done, _ = env.step(action)
    obs = obs_all['image']
    num_step += 1
    if done:
        print('Step:', num_step, 'reward: ', reward)
    if renderer.window is None:
        break
