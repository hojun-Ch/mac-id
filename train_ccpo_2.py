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
from eval_ccpo_2 import evaluation
from utils import state_engineering, obs_to_reward, obs_to_reward_coll_smoothed, obs_to_single_reward, make_observation, make_env, check_success_rate

import wandb

if __name__ == '__main__':
    
    # get and save args
    args = parser.parse_args()
    args.algo = "MAC-ID-conditioned"
    
    wandb.init(project="pedsim_v2", reinit=True, entity="hojun-chung")
    wandb.run.name = args.name
    
    wandb.config.update(args)
    
    # make model directory
    if not os.path.isdir(args.model_path + args.name):
        os.mkdir(args.model_path + args.name)
        
    agent = CCPO(args)
 
    # make train env
    env = make_env(args, "train_" + args.env_name, args.seed, 0)
    eval_env = make_env(args, "eval_" + args.env_name, 12345, 1)
    
    print("total episode:", args.max_step // args.rollout_length)
    
    best_return = -1000

    for j in range(args.max_step // args.rollout_length):
        
        agent.train_mode()
        
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
        pg_losses, clip_fraction,  i_value_loss, n_value_loss, g_value_loss = agent.update_policy()
        
        # reset buffer 
        agent.rollout_buffer.reset()
        
        wandb.log({
            'train_return': train_global_return,
            'train_collision': train_collision,
            'train_success_rate': success_rate
        }, step = j)
        
        # evaluation
        if j % args.eval_frequency == 0:
            print("now on evaluation")
            
            agent.eval_mode()
            
            eval_return, eval_coll, eval_sr = evaluation(args, agent, eval_env)

            
            
            if eval_return > best_return:
                best_return = eval_return
                agent.save_model(args.model_path + args.name + "/model_best.pt")
            
            wandb.log({
                'eval_returns': eval_return,
                'eval_collisions': eval_coll,
                "eval_success_rate": eval_sr
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
    eval_env.close()



