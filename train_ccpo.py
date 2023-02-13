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
from eval_ccpo import eval_bottleneck, eval_crossing, eval_dense, eval_random
from utils import state_engineering,obs_to_reward, obs_to_reward_coll_smoothed, make_observation, make_env, check_success_rate

import wandb

if __name__ == '__main__':
    
    # get and save args
    args = parser.parse_args()
    args.algo = "MAC_ID_conditioned"
    wandb.init(project="pedsim_lidar", reinit=True, entity="hojun-chung")
    wandb.run.name = args.name
    
    wandb.config.update(args)
    
    # make model directory
    if not os.path.isdir(args.model_path + args.name):
        os.mkdir(args.model_path + args.name)
        
    
    agent = CCPO(args)
    
    # make train env
    env = make_env(args, "train_stage", args.seed, 0)
    bottleneck_env = make_env(args, "eval_bottleneck", 12345, 1)
    crossing_env = make_env(args, "eval_crossing", 12345, 2)
    dense_env = make_env(args, "eval_dense", 12345, 3)
    random_env = make_env(args, "eval_random", 12345, 4)
   
    print("total episode:", args.max_step // args.rollout_length)
    
    best_return = -1000

    for j in range(args.max_step // args.rollout_length):
        
        agent.train_mode()
        
        # env_path = args.home_path + args.env_path
        
        # channel = EngineConfigurationChannel()

        # unity_env = UE(file_name = env_path + "train_stage.x86_64", seed=np.random.randint(100000), side_channels=[channel], no_graphics=not args.rendering, worker_id = args.worker)

        # env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)
        # channel.set_configuration_parameters(time_scale = 2.0)
        
        raw_obs = env.reset()
        obs = make_observation(raw_obs[0], args.map_length, args.map_width, args.num_ped, args.obs_dim, args.dummy_index, args.neighbor_distance)

        print("episode num: ", j)
        print("collecting rollout.....")

        # collect rollout
        train_global_return = 0
        train_collision = 0
        lcf = np.random.uniform(-math.pi / 2, math.pi / 2, args.num_ped)

        
        for i in range(args.rollout_length):
            prev_obs = obs
            new_prev_obs = state_engineering(prev_obs, args.map_length, args.map_width, args.num_ped, args.obs_dim)

            with torch.no_grad():
                action, log_prob, value, n_value, g_value = agent.act(torch.from_numpy(new_prev_obs), lcf)
            raw_obs, __, __, __ = env.step(action.reshape(-1))
            
            obs = make_observation(raw_obs[0], args.map_length, args.map_width, args.num_ped, args.obs_dim, args.dummy_index, args.neighbor_distance)

            if args.smooth_cost:
                reward, n_reward, g_reward, global_reward_wo_coll, global_coll = obs_to_reward_coll_smoothed(obs, prev_obs, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance, args.sparse_reward)
            else:
                reward, n_reward, g_reward, global_reward_wo_coll, global_coll = obs_to_reward(obs, prev_obs, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance, args.sparse_reward)
            
            train_global_return += global_reward_wo_coll
            train_collision +=  global_coll
            if i == 0:
                agent.rollout_buffer.add(new_prev_obs, action, reward, n_reward, g_reward, np.array([1] * args.num_ped), value, n_value, g_value, log_prob, lcf)
            else:
                agent.rollout_buffer.add(new_prev_obs, action, reward, n_reward, g_reward, np.array([0] * args.num_ped), value, n_value, g_value, log_prob, lcf)
        print("episode end!")
        
        #  compute advantage and return
        success_rate = check_success_rate(obs, args.map_length, args.map_width)
        with torch.no_grad():
            last_value, last_n_value, last_g_value = agent.get_values(torch.from_numpy(state_engineering(obs, args.map_length, args.map_width, args.num_ped, args.obs_dim)),lcf)
        agent.rollout_buffer.compute_returns_and_advantage(last_value, last_n_value, last_g_value, 0)

        print("updating....")
        
        # update policy 
        pg_losses, clip_fraction, i_value_loss, n_value_loss, g_value_loss = agent.update_policy()
        
        # reset buffer 
        agent.rollout_buffer.reset()
        
        wandb.log({
            'train_return': train_global_return,
            'train_collision': train_collision,
            'train_success_rate': success_rate
        }, step=j)
        
        # evaluation
        if j % args.eval_frequency == 0:
            print("now on evaluation")
            
            agent.eval_mode()
            
            bottleneck_return, bottleneck_coll, bottleneck_sr = eval_bottleneck(args, agent, bottleneck_env)
            crossing_return, crossing_coll, crossing_sr = eval_crossing(args, agent, crossing_env)
            dense_return, dense_coll, dense_sr = eval_dense(args, agent, dense_env)
            random_return, random_coll, random_sr = eval_random(args, agent, random_env)
            
            if bottleneck_return + crossing_return + dense_return + random_return > best_return:
                best_return = bottleneck_return + crossing_return + dense_return + random_return
                agent.save_model(args.model_path + args.name + "/model_best.pt")
            
            wandb.log({
                'bottleneck eval returns': bottleneck_return,
                'bottleneck_eval_collisions': bottleneck_coll,
                "bottleneck_eval_success_rate": bottleneck_sr,
                'crossing eval returns': crossing_return,
                "crossing_eval_collisions": crossing_coll,
                "crossing_eval_success_rate": crossing_sr,
                'dense eval returns': dense_return,
                'dense_eval_collisions': dense_coll,
                "dense_eval_success_rate": dense_sr,
                'random_eval_returns': random_return,
                'random_eval_collisions': random_coll,
                "random_eval_success_rate": random_sr
            }, step=j)
        
        # logging
        
        if j % args.archive_frequency == 0:
            agent.save_model(args.model_path + args.name + "/model_%05d.pt"%j)
        
        # for i in range(len(pg_losses)):
        #     wandb.log({
        #         'policy losses': pg_losses[i],
        #         'clip fractions': clip_fraction[i],
        #         'independent value losses': i_value_loss[i]
        #     })

        

    env.close()
    bottleneck_env.close()
    crossing_env.close()
    random_env.close()








