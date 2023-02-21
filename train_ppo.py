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
from agents.ipo import IPO
from eval_ipo_2 import evaluation
from utils import state_engineering_ppo, obs_to_single_reward, make_observation, make_env, check_success_rate, shortest_distance

import wandb

if __name__ == '__main__':
    
    # get and save args
    args = parser.parse_args()
    # args.algo = "IPO"
    
    wandb.init(project="pedsim_v2", reinit=True, entity="hojun-chung")
    wandb.run.name = args.name
    
    wandb.config.update(args)
    
    # make model directory
    if not os.path.isdir(args.model_path + "easy/" + args.name):
        os.mkdir(args.model_path + "easy/" + args.name)
    
    # change args.num_ped if env is "dense"
    if args.env_name == "dense":
        args.num_ped = 200
        
    agent = IPO(args)
 
    # make train env
    env = make_env(args, "train_" + args.env_name, args.seed, args.worker)
    # eval_env = make_env(args, "eval_" + args.env_name, 12345, args.worker + 1)
    
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
        for i in range(args.rollout_length):
            prev_obs = obs
            new_prev_obs = state_engineering_ppo(prev_obs, args.map_length, args.map_width, args.num_ped, args.obs_dim)

            action, log_prob, value = agent.act(torch.from_numpy(new_prev_obs))
            raw_obs, __, __, __ = env.step(action.reshape(-1))
            # raw_obs, __, __, __ = env.step(np.array([1.0, 1.0]))            
            
            obs = make_observation(raw_obs[0], args.map_length, args.map_width, args.num_ped, args.obs_dim, args.dummy_index, args.neighbor_distance)
            reward, g_reward, global_reward_wo_coll, global_coll = obs_to_single_reward(obs, prev_obs, args.map_length, args.map_width, args.num_ped, args.coll_penalty, args.neighbor_distance, args.sparse_reward)
            
            train_global_return += global_reward_wo_coll
            train_collision +=  global_coll
            
            if i == 0:
                agent.rollout_buffer.add(new_prev_obs, action, reward, reward, g_reward, np.array([1] * args.num_ped), value, value, value, log_prob, np.array([0.0] * args.num_ped))
            else:
                agent.rollout_buffer.add(new_prev_obs, action, reward, reward, g_reward, np.array([0] * args.num_ped), value, value, value, log_prob, np.array([0.0] * args.num_ped))
        
        print("episode end!")
        
        #  compute advantage and return
        success_rate, __ = check_success_rate(obs, args.map_length, args.map_width)
        last_value = agent.get_values(torch.from_numpy(state_engineering_ppo(obs, args.map_length, args.map_width, args.num_ped, args.obs_dim)))
        agent.rollout_buffer.compute_returns_and_advantage(last_value, last_value, last_value, 0)

        print("updating....")
        
        # update policy 
        pg_losses, clip_fraction, i_value_loss = agent.update_policy()
        
        
        # reset buffer 
        agent.rollout_buffer.reset()
        
        wandb.log({
            'train_return': train_global_return,
            'train_collision': train_collision,
            'train_success_rate': success_rate
        })
        
        # # evaluation
        # if j % args.eval_frequency == 0:
        #     print("now on evaluation")
            
        #     agent.eval_mode()
            
        #     eval_return, eval_coll, eval_sr, eval_efficiency = evaluation(args, agent, eval_env)

            
            
        #     if eval_return > best_return:
        #         best_return = eval_return
        #         agent.save_model(args.model_path + "easy/" + args.name + "/model_best.pt")
            
        #     wandb.log({
        #         'eval_returns': eval_return,
        #         'eval_collisions': eval_coll,
        #         "eval_success_rate": eval_sr,
        #         "eval_path_efficiency":eval_efficiency
        #     }, step=j)
        
        # logging
        
        if j % args.archive_frequency == 0:
            agent.save_model(args.model_path + "easy/" + args.name + "/model_%05d.pt"%j)
        
        for i in range(len(pg_losses)):
            wandb.log({
                'policy losses': pg_losses[i],
                'clip fractions': clip_fraction[i],
                'independent value losses': i_value_loss[i]
            })

        

    env.close()
    # eval_env.close()



