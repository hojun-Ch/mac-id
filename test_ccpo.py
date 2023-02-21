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
from agents.ccpo import CCPO

import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os

from utils import obs_to_global_reward, make_observation, check_success_rate, state_engineering, shortest_distance, make_env

def evaluation(args, agent, env, lcf):
    total_return = np.zeros(5)
    total_coll = np.zeros(5)
    total_success_rate =  np.zeros(5)
    total_path_efficiency = np.zeros(5)
    total_success_rate_wocoll =  np.zeros(5)
    for seed in range(5):
        raw_obs = env.reset()
        obs = make_observation(raw_obs[0],args.map_length, args.map_width, args.num_ped, args.obs_dim, args.dummy_index, args.neighbor_distance)
        l2_distance = shortest_distance(obs, args.map_length, args.map_width, args.num_ped)
        total_distance = np.zeros(args.num_ped)
        episode_return = 0
        episode_coll = 0
        check_collision = np.zeros(args.num_ped)
        for i in range(1000):
            prev_obs = obs
            new_prev_obs = state_engineering(prev_obs, args.map_length, args.map_width, args.num_ped, args.obs_dim)
            action, __, __, __, __ = agent.act(torch.from_numpy(new_prev_obs), lcf)
            raw_obs, __, __, __ = env.step(action.reshape(-1))
            
            obs = make_observation(raw_obs[0],args.map_length, args.map_width, args.num_ped, args.obs_dim, args.dummy_index, args.neighbor_distance)
            g_reward, g_coll, collision, move = obs_to_global_reward(obs, prev_obs, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance)
            total_distance += move
            episode_return += g_reward
            episode_coll += g_coll
            check_collision += collision
        success_rate, success = check_success_rate(obs, args.map_length, args.map_width)
        success[check_collision > 0] = 0
        success_rate_wocoll = np.mean(success)
        path_efficiency = np.mean(l2_distance[success] / total_distance[success])
        
        total_return[seed] = episode_return
        total_coll[seed] = episode_coll
        total_success_rate[seed] = success_rate
        total_success_rate_wocoll[seed] = success_rate_wocoll
        total_path_efficiency[seed] = path_efficiency
        
    return np.mean(total_return), np.mean(total_coll), np.mean(total_success_rate), np.mean(total_success_rate_wocoll), np.mean(total_path_efficiency)

if __name__ == '__main__':
    
    np.random.seed(0)
    torch.manual_seed(0) 
    
    # get and save args
    args = parser.parse_args()
    
    if args.env_name == "dense" or args.env_name == "easydense" or args.env_name == "sparsedense":
        args.num_ped = 200
    if args.env_name == "100dense" or args.env_name == "100easydense":
        args.num_ped = 100
    if args.env_name == "sparse":
        args.num_ped = 25
        
    agent = CCPO(args)


    env = make_env(args, "eval_" + args.env_name, 0, args.worker)

    # load trained agent
    seed_list = [3, 34, 89, 233, 315, 987, 1597]
    score, collision, success_rate, success_rate_wocoll, path_efficiency = np.zeros(7), np.zeros(7), np.zeros(7), np.zeros(7), np.zeros(7)
    
    lcf_list=[]
    for i in range(6):
        lcf_list.append(np.random.uniform(-math.pi / 2, - math.pi / 3, args.num_ped) + i * math.pi / 6)
    lcf_list.append(np.random.uniform(0, math.pi / 2, args.num_ped))
    lcf_list.append(np.random.uniform(0, math.pi / 3, args.num_ped))
    lcf_list.append(np.ones(args.num_ped) * math.pi / 6)
    lcf_list.append(np.ones(args.num_ped) * math.pi / 4)
    lcf_list.append(np.ones(args.num_ped) * math.pi / 3)
    lcf_name = ["-90 ~ -60", "-60 ~ -30", "-30 ~ 0", "0 ~ 30", "30 ~ 60", "60 ~ 90", "0 ~ 90", "0 ~ 60", "30", "45", "60"]
    
    for idx in range(len(lcf_list)):
    # for idx in [4,9,10]:
        for i in range(len(seed_list)):
            print("seed:", seed_list[i])
            
            agent.load_ckpt(args.model_path + "easy/" + args.name +"_seed_"+ str(seed_list[i]) + "/model_best.pt")
            agent.eval_mode()
            
            score[i], collision[i], success_rate[i], success_rate_wocoll[i],path_efficiency[i] = evaluation(args, agent, env, lcf_list[idx])

        print('-' * 20)
        print(args.env_name + "_" + args.algo + '_result')
        print("lcf:", lcf_name[idx])
        print("score:", np.mean(score) / 20)
        print("collision:", np.mean(collision))
        print("success_rate:", np.mean(success_rate))
        print("success_rate w/o collision:", np.mean(success_rate_wocoll))
        print("path_efficiency:", np.mean(path_efficiency))
        print('-' * 20)
    env.close()

