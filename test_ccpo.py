import cv2
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import json

from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper

import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
from arguments import parser
from agents.ccpo import CCPO
from utils import obs_to_global_reward, make_observation, make_env, check_success_rate, state_engineering

import wandb

def test_bottleneck(args, agent, env):
    total_return = []
    total_coll = []
    total_sr = []
    
    # lcf_list = [np.random.uniform(0, math.pi/3, 51)]
    lcf_list = []
    for i in range(19):
        lcf_list.append(np.ones(51) * (-math.pi / 2 + i * math.pi / 18))
    lcf_list.append(np.random.uniform(0, math.pi / 2, 51))
    lcf_list.append(np.random.uniform(0, math.pi / 6, 51))
    lcf_list.append(np.random.uniform(math.pi / 6, math.pi / 3, 51))
    lcf_list.append(np.random.uniform(-math.pi / 6, math.pi / 3, 51))
    
    for lcf in lcf_list:
        raw_obs = env.reset()
        obs = make_observation(raw_obs[0],args.map_length, args.map_width, 51, args.obs_dim, args.dummy_index, args.neighbor_distance)
        episode_return = 0
        episode_coll = 0
        for i in range(1000):
            prev_obs = obs
            new_prev_obs = state_engineering(prev_obs, args.map_length, args.map_width, 51, args.obs_dim)
            action, log_prob, value, n_value, g_value  = agent.act(torch.from_numpy(new_prev_obs), lcf)
            raw_obs, __, __, __ = env.step(action.reshape(-1))
            
            obs = make_observation(raw_obs[0],args.map_length, args.map_width, 51, args.obs_dim, args.dummy_index, args.neighbor_distance)
            g_reward, g_coll = obs_to_global_reward(obs, prev_obs, args.map_length, args.map_width, 51, args.coll_penalty, args.neighbor_distance)
            episode_return += g_reward
            episode_coll += g_coll
        success_rate = check_success_rate(obs, args.map_length, args.map_width)
        total_return.append(episode_return)
        total_coll.append(episode_coll)
        total_sr.append(success_rate)
        
    total_return = np.array(total_return)
    total_coll = np.array(total_coll)
    total_sr = np.array(total_sr)
    return total_return, total_coll, total_sr

def test_crossing(args, agent, env):
    total_return = []
    total_coll = []
    total_sr = []
    
    # lcf_list = [np.random.uniform(0, math.pi/3, 51)]
    lcf_list = []
    for i in range(19):
        lcf_list.append(np.ones(51) * (-math.pi / 2 + i * math.pi / 18))
    lcf_list.append(np.random.uniform(0, math.pi / 2, 51))
    lcf_list.append(np.random.uniform(0, math.pi / 6, 51))
    lcf_list.append(np.random.uniform(math.pi / 6, math.pi / 3, 51))
    lcf_list.append(np.random.uniform(-math.pi / 6, math.pi / 3, 51))
    
    for lcf in lcf_list:
        raw_obs = env.reset()
        obs = make_observation(raw_obs[0],args.map_length, args.map_width, 51, args.obs_dim, args.dummy_index, args.neighbor_distance)
        episode_return = 0
        episode_coll = 0
        for i in range(1000):
            prev_obs = obs
            new_prev_obs = state_engineering(prev_obs, args.map_length, args.map_width, 51, args.obs_dim)
            action, log_prob, value, n_value, g_value  = agent.act(torch.from_numpy(new_prev_obs), lcf)
            raw_obs, __, __, __ = env.step(action.reshape(-1))
            
            obs = make_observation(raw_obs[0],args.map_length, args.map_width, 51, args.obs_dim, args.dummy_index, args.neighbor_distance)
            g_reward, g_coll = obs_to_global_reward(obs, prev_obs, args.map_length, args.map_width, 51, args.coll_penalty, args.neighbor_distance)
            episode_return += g_reward
            episode_coll += g_coll
        success_rate = check_success_rate(obs, args.map_length, args.map_width)
        
        total_return.append(episode_return)
        total_coll.append(episode_coll)
        total_sr.append(success_rate)
    total_return = np.array(total_return)
    total_coll = np.array(total_coll)
    total_sr = np.array(total_sr)
    return total_return, total_coll, total_sr

def test_dense(args, agent, env):
    total_return = []
    total_coll = []
    total_sr = []
    
    # lcf_list = [np.random.uniform(0, math.pi/3, 200)]
    lcf_list = []
    for i in range(19):
        lcf_list.append(np.ones(200) * (-math.pi / 2 + i * math.pi / 18))
    lcf_list.append(np.random.uniform(0, math.pi / 2, 200))
    lcf_list.append(np.random.uniform(0, math.pi / 6, 200))
    lcf_list.append(np.random.uniform(math.pi / 6, math.pi / 3, 200))
    lcf_list.append(np.random.uniform(-math.pi / 6, math.pi / 3, 200))

    for lcf in lcf_list:
        raw_obs = env.reset()
        obs = make_observation(raw_obs[0],args.map_length, args.map_width, 200, args.obs_dim, args.dummy_index, args.neighbor_distance)
        episode_return = 0
        episode_coll = 0
        for i in range(1000):
            prev_obs = obs
            new_prev_obs = state_engineering(prev_obs, args.map_length, args.map_width, 200, args.obs_dim)
            action, log_prob, value,  n_value, g_value  = agent.act(torch.from_numpy(new_prev_obs), lcf)
            raw_obs, __, __, __ = env.step(action.reshape(-1))
            
            obs = make_observation(raw_obs[0],args.map_length, args.map_width, 200, args.obs_dim, args.dummy_index, args.neighbor_distance)
            g_reward, g_coll = obs_to_global_reward(obs, prev_obs, args.map_length, args.map_width, 200, args.coll_penalty, args.neighbor_distance)
            episode_return += g_reward
            episode_coll += g_coll
        success_rate = check_success_rate(obs, args.map_length, args.map_width)
        
        total_return.append(episode_return)
        total_coll.append(episode_coll)
        total_sr.append(success_rate)
    
    total_return = np.array(total_return)
    total_coll = np.array(total_coll)
    total_sr = np.array(total_sr)
    return total_return, total_coll, total_sr

def test_random(args, agent, env):
    
    total_return = []
    total_coll = []
    total_sr = []

    # lcf_list = [np.random.uniform(0, math.pi/3, 51)]
    lcf_list=[]
    for i in range(19):
        lcf_list.append(np.ones(51) * (-math.pi / 2 + i * math.pi / 18))
    lcf_list.append(np.random.uniform(0, math.pi / 2, 51))
    lcf_list.append(np.random.uniform(0, math.pi / 6, 51))
    lcf_list.append(np.random.uniform(math.pi / 6, math.pi / 3, 51))
    lcf_list.append(np.random.uniform(-math.pi / 6, math.pi / 3, 51))

    for lcf in lcf_list:
        raw_obs = env.reset()
        obs = make_observation(raw_obs[0],args.map_length, args.map_width, 51, args.obs_dim, args.dummy_index, args.neighbor_distance)
        episode_return = 0
        episode_coll = 0
        for i in range(1000):
            prev_obs = obs
            new_prev_obs = state_engineering(prev_obs, args.map_length, args.map_width, 51, args.obs_dim)
            action, log_prob, value,  n_value, g_value  = agent.act(torch.from_numpy(new_prev_obs), lcf)
            raw_obs, __, __, __ = env.step(action.reshape(-1))
            
            obs = make_observation(raw_obs[0],args.map_length, args.map_width, 51, args.obs_dim, args.dummy_index, args.neighbor_distance)
            g_reward, g_coll = obs_to_global_reward(obs, prev_obs, args.map_length, args.map_width, 51, args.coll_penalty, args.neighbor_distance)
            episode_return += g_reward
            episode_coll += g_coll
        success_rate = check_success_rate(obs, args.map_length, args.map_width)
        
        total_return.append(episode_return)
        total_coll.append(episode_coll)
        total_sr.append(success_rate)
    total_return = np.array(total_return)
    total_coll = np.array(total_coll)
    total_sr = np.array(total_sr)
    return total_return, total_coll, total_sr


if __name__ == '__main__':
    
    # get and save args
    args = parser.parse_args()
    args.algo = "MAC_ID_conditioned"

    bottleneck_env = make_env(args, "eval_bottleneck", 12345, 1)
    crossing_env = make_env(args, "eval_crossing", 12345, 2)
    dense_env = make_env(args, "eval_dense", 12345, 3)
    random_env = make_env(args, "eval_random", 12345, 4)
    
    # load trained agent
    agent = CCPO(args)
    seed_list = [315,987,1597]
    bottleneck_return, bottleneck_coll, bottleneck_sr = np.zeros((3,23)), np.zeros((3,23)), np.zeros((3,23))
    crossing_return, crossing_coll, crossing_sr = np.zeros((3,23)), np.zeros((3,23)), np.zeros((3,23))
    dense_return, dense_coll, dense_sr = np.zeros((3,23)), np.zeros((3,23)), np.zeros((3,23))
    random_return, random_coll, random_sr = np.zeros((3,23)), np.zeros((3,23)), np.zeros((3,23))
    
    for i in range(len(seed_list)):
        print("seed:", seed_list[i])
        agent.load_ckpt(args.model_path + args.name +"_seed_"+ str(seed_list[i]) + "/model_best.pt")
        agent.eval_mode()
        
    
        bottleneck_return[i], bottleneck_coll[i], bottleneck_sr[i] = test_bottleneck(args, agent, bottleneck_env)
        crossing_return[i], crossing_coll[i], crossing_sr[i] = test_crossing(args, agent, crossing_env)
        dense_return[i], dense_coll[i], dense_sr[i] = test_dense(args, agent, dense_env)
        random_return[i], random_coll[i], random_sr[i] = test_random(args, agent, random_env)
        
    bottleneck_env.close()
    crossing_env.close()
    random_env.close()

    print('-' * 20)
    print('bottleneck result')
    print(np.mean(bottleneck_return, axis=0))
    print(np.mean(bottleneck_coll, axis=0))
    print(np.mean(bottleneck_sr, axis=0))
    print('-' * 20)
    
    print('-' * 20)
    print('crossing result')
    print(np.mean(crossing_return, axis=0))
    print(np.mean(crossing_coll, axis=0))
    print(np.mean(crossing_sr, axis=0))
    print('-' * 20)
    
    print('-' * 20)
    print('dense result')
    print(np.mean(dense_return, axis=0))
    print(np.mean(dense_coll, axis=0))
    print(np.mean(dense_sr, axis=0))
    print('-' * 20)
    
    print('-' * 20)
    print('random result')
    print(np.mean(random_return, axis=0))
    print(np.mean(random_coll, axis=0))
    print(np.mean(random_sr, axis=0))
    print('-' * 20)







