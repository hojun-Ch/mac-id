import cv2
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import random
import json
from arguments import parser

from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper

import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os

from utils import obs_to_global_reward, make_observation, check_success_rate, state_engineering

def evaluation(args, agent, env):

    raw_obs = env.reset()
    obs = make_observation(raw_obs[0],args.map_length, args.map_width, args.num_ped, args.obs_dim, args.dummy_index, args.neighbor_distance)
    episode_return = 0
    episode_coll = 0
    
    for i in range(1000):
        prev_obs = obs
        new_prev_obs = state_engineering(prev_obs, args.map_length, args.map_width, args.num_ped, args.obs_dim)
        action, log_prob, value = agent.act(torch.from_numpy(new_prev_obs))
        raw_obs, __, __, __ = env.step(action.reshape(-1))
        
        obs = make_observation(raw_obs[0],args.map_length, args.map_width, args.num_ped, args.obs_dim, args.dummy_index, args.neighbor_distance)
        g_reward, g_coll = obs_to_global_reward(obs, prev_obs, args.map_length, args.map_width, 51, args.coll_penalty, args.neighbor_distance)
        episode_return += g_reward
        episode_coll += g_coll
    success_rate = check_success_rate(obs, args.map_length, args.map_width)
    return episode_return, episode_coll, success_rate
