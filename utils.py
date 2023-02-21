import cv2
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel


red_color = (1.0,0.0,0.0)
green_color = (0.0,1.0,0.0)
blue_color = (0.0,0.0,1.0)
white_color = (1.0,1.0,1.0)
black_color = (0.0,0.0,0.0)


def dist_to_reward(distance, prev_distance, max_distance, neighbor_distance, sparse=False):
    if sparse:
        reward = -0.01
        
        if distance < neighbor_distance:
            reward = (neighbor_distance - distance) / neighbor_distance
        
        if distance < 0.5:
            reward = 2.0
        
        return reward
        
    else:
        reward = (prev_distance - distance)
        
        if distance < neighbor_distance:
            reward += (neighbor_distance - distance) / neighbor_distance
            
        if distance < 0.5:
            reward = 2.0
            
        return reward
        

def make_observation(raw_obs, l, w, num_ped, obs_dim, index_offset, neighbor_distance):
    
    obs = np.array(raw_obs[index_offset:], dtype = np.float32)
    obs = obs.reshape((num_ped, 43))
    dmean = np.array([0,0,180,0,0,0,0] + [0,180,0] * 12, dtype=np.float32)
    normalize_factor = np.array([l//2, w//2, 180, l//2, w//2, 5, 1] + [1,180,5] * 12, dtype=np.float32)
    obs = (obs-dmean) / normalize_factor
    obs[:,6] = obs[:,6] > 0
    
    return obs

def make_observation_mf(raw_obs, l, w, num_ped, obs_dim, index_offset, neighbor_distance):
    neighbor_obs = np.zeros((num_ped, 43))
    obs = np.array(raw_obs[index_offset:], dtype = np.float32)
    obs = obs.reshape((num_ped, 43))
    dmean = np.array([0,0,180,0,0,0,0] + [0,180,0] * 12, dtype=np.float32)
    normalize_factor = np.array([l//2, w//2, 180, l//2, w//2, 5, 1] + [1,180,5] * 12, dtype=np.float32)
    obs = (obs-dmean) / normalize_factor
    obs[:,6] = obs[:,6] > 0
    
    location = obs[:,:2] * np.array([l//2, w//2])
    
    for i in range(num_ped):
        location_difference = location - location[i]
        distance = np.sum(location_difference ** 2, axis=1)
        neighbor_index = distance < (neighbor_distance ** 2)
        neighbor_obs[i] = np.mean(obs[neighbor_index], axis=0)
        
        
    
    return obs, neighbor_obs

def state_engineering(obs, l, w, num_ped, obs_dim):
    # distance
    new_obs = np.zeros((num_ped, obs_dim), dtype=np.float32)
    new_obs[:,0] = np.sqrt((obs[:,0] - obs[:,3])**2 + (obs[:,1] - obs[:,4])**2)
    
    # heading
    goal_direction = (np.arctan2(obs[:,0] - obs[:,3], obs[:,1] - obs[:,4]))
    goal_direction[goal_direction < 0] += math.pi * 2
    
    # goal_heading = goal_direction - obs[:,2] * math.pi / 180
    # goal_heading[goal_heading < 0] += math.pi * 2
    
    goal_heading = goal_direction - obs[:,2] * math.pi
    
    new_obs[:,1] = goal_heading / (2 * math.pi)
    
    new_obs[:,2:] = obs[:,7:]
    return new_obs

def state_engineering_ppo(obs, l, w, num_ped, obs_dim):
    # distance
    new_obs = np.zeros((num_ped, obs_dim), dtype=np.float32)
    new_obs[:,0] = np.sqrt((obs[:,0] - obs[:,3])**2 + (obs[:,1] - obs[:,4])**2)
    
    # heading
    goal_direction = (np.arctan2(obs[:,3] - obs[:,0], obs[:,4] - obs[:,1]))
    goal_direction[goal_direction < 0] += math.pi * 2
    goal_heading = goal_direction - obs[:,2] * math.pi
    # goal_heading[goal_heading < 0] += math.pi * 2
    new_obs[:,1] = goal_heading
    # new_obs = obs[:,:5]
    
    new_obs[:,2:] = obs[:,7:]
    return new_obs

def obs_to_reward(obs, prev_obs, l, w, num_ped, coll_penalty, neighbor_distance, sparse):
    reward = np.zeros(num_ped)
    g_reward = np.ones(num_ped)
    n_reward = np.zeros(num_ped)
    neighbor_num = np.zeros(num_ped)
    no_coll_reward = np.ones(num_ped)
    collision = np.zeros(num_ped)
    
    max_distance = math.sqrt(l**2 + w**2)
    
    for i in range(num_ped):
        
        x_0 = prev_obs[i][0] * (l//2)
        z_0 = prev_obs[i][1] * (w//2)
        
        x = obs[i][0] * (l//2)
        z = obs[i][1] * (w//2)
        coll = obs[i][6]

        goal_x = obs[i][3] * (l//2)
        goal_z = obs[i][4] * (w//2)
        
        prev_distance = math.sqrt((goal_x - x_0)**2 + (goal_z - z_0) ** 2) 
        distance_to_goal = math.sqrt((goal_x - x)**2 + (goal_z - z) ** 2)        
        
        reward[i] = dist_to_reward(distance_to_goal, prev_distance, max_distance, neighbor_distance, sparse) - coll * coll_penalty
        no_coll_reward[i] = dist_to_reward(distance_to_goal, prev_distance, max_distance, neighbor_distance)
        collision[i] = coll
    
    location = obs[:,:2] * np.array([l//2, w//2])
    g_reward *= reward.sum() / num_ped
    
    for i in range(num_ped):
        location_difference = location - location[i]
        distance = np.sum(location_difference ** 2, axis=1)
        neighbor_index = distance < (neighbor_distance ** 2)
        neighbor_index[i] = 0
        if np.shape(reward[neighbor_index])[0] == 0:
            n_reward[i] = g_reward[0]
        else:
            n_reward[i] = np.mean(reward[neighbor_index])
        
    # n_reward = n_reward / (neighbor_num +1e-8)
    # print(neighbor_num)
    # n_reward[neighbor_num == 0] = g_reward[0]
    
    global_reward_wo_coll = no_coll_reward.sum() / num_ped
    g_coll = collision.sum() / num_ped
    return reward, n_reward, g_reward, global_reward_wo_coll, g_coll

def obs_to_single_reward(obs, prev_obs, l, w, num_ped, coll_penalty, neighbor_distance, sparse):
    reward = np.zeros(num_ped)
    g_reward = np.ones(num_ped)
    no_coll_reward = np.ones(num_ped)
    collision = np.zeros(num_ped)
    
    max_distance = math.sqrt(l**2 + w**2)
    
    for i in range(num_ped):
        
        x_0 = prev_obs[i][0] * (l//2)
        z_0 = prev_obs[i][1] * (w//2)
        
        x = obs[i][0] * (l//2)
        z = obs[i][1] * (w//2)

        coll = obs[i][6]
        goal_x = obs[i][3] * (l//2)
        goal_z = obs[i][4] * (w//2)
        
        prev_distance = math.sqrt((goal_x - x_0)**2 + (goal_z - z_0) ** 2) 
        distance_to_goal = math.sqrt((goal_x - x)**2 + (goal_z - z) ** 2)        
        
        reward[i] = dist_to_reward(distance_to_goal, prev_distance, max_distance, neighbor_distance, sparse) - coll * coll_penalty
        no_coll_reward[i] = dist_to_reward(distance_to_goal, prev_distance, max_distance, neighbor_distance)
        collision[i] = coll

        
    g_reward *= reward.sum() / num_ped
    
    global_reward_wo_coll = no_coll_reward.sum() / num_ped
    g_coll = collision.sum() / num_ped
    return reward, g_reward, global_reward_wo_coll, g_coll


def obs_to_global_reward(obs, prev_obs, l, w, num_ped, coll_penalty, neighbor_distance):
    reward = np.zeros(num_ped)
    collision = np.zeros(num_ped)
    max_distance = math.sqrt(l**2 + w**2)
    moving_distance = np.zeros(num_ped)
    for i in range(num_ped):
        
        x_0 = prev_obs[i][0] * (l//2)
        z_0 = prev_obs[i][1] * (w//2)
        
        x = obs[i][0] * (l//2)
        z = obs[i][1] * (w//2)
        coll = obs[i][6]

        goal_x = obs[i][3] * (l//2)
        goal_z = obs[i][4] * (w//2)
        
        prev_distance = math.sqrt((goal_x - x_0)**2 + (goal_z - z_0) ** 2) 
        distance_to_goal = math.sqrt((goal_x - x)**2 + (goal_z - z) ** 2)     
        moving_distance[i] = math.sqrt((x - x_0)**2 + (z - z_0) ** 2) 
        
        reward[i] = dist_to_reward(distance_to_goal, prev_distance, max_distance, neighbor_distance)
        if reward[i] == 2.0:
            moving_distance[i] = 0
            
        collision[i] = coll
        
    
    g_reward = reward.sum() / num_ped
    g_coll = collision.sum() / num_ped
    
    return g_reward, g_coll, collision, moving_distance

def obs_to_reward_coll_smoothed(obs, prev_obs, l, w, num_ped, coll_penalty, neighbor_distance, sparse):
    reward = np.zeros(num_ped)
    g_reward = np.ones(num_ped)
    n_reward = np.zeros(num_ped)
    neighbor_num = np.ones(num_ped)
    no_coll_reward = np.ones(num_ped)
    collision = np.zeros(num_ped)
    
    max_distance = math.sqrt(l**2 + w**2)
    
    for i in range(num_ped):
        
        x_0 = prev_obs[i][0] * (l//2)
        z_0 = prev_obs[i][1] * (w//2)
        
        x = obs[i][0] * (l//2)
        z = obs[i][1] * (w//2)
        coll = obs[i][6]

        goal_x = obs[i][3] * (l//2)
        goal_z = obs[i][4] * (w//2)
        
        prev_distance = math.sqrt((goal_x - x_0)**2 + (goal_z - z_0) ** 2) 
        distance_to_goal = math.sqrt((goal_x - x)**2 + (goal_z - z) ** 2)        
        
        no_coll_reward[i] = dist_to_reward(distance_to_goal, prev_distance, max_distance, neighbor_distance, sparse)
        
        collision[i] = coll
        
        cost = 0
        min_distance = 1000
        all_x = obs[:,0]
        all_z = obs[:,1]
   
        x_diff = x - all_x
        z_diff = z - all_z
        
        all_dist = np.sqrt(x_diff**2 + z_diff**2)
        all_dist[i] = 1000
        min_distance = np.min(all_dist)
        
        if min_distance < 2.0:
            cost = coll_penalty * ((2.0 / min_distance) ** 2 - 1) / 3
        
        reward[i] = no_coll_reward[i] - cost
        no_coll_reward[i] = dist_to_reward(distance_to_goal, prev_distance, max_distance, neighbor_distance)
             
        
    for i in range(num_ped):
        n_reward[i] += reward[i]

        for j in range(i+1, num_ped):
            x1 = obs[i][0] * (l//2)
            z1 = obs[i][1] * (w//2)
            x2 = obs[j][0] * (l//2)
            z2 = obs[j][1] * (w//2)
            
            if math.sqrt((x1 - x2)**2 + (z1 - z2) ** 2) < neighbor_distance:
                n_reward[i] += reward[j]
                n_reward[j] += reward[i]
                neighbor_num[i] += 1
                neighbor_num[j] += 1
                            
    n_reward = n_reward / (neighbor_num +1e-8)
    
    g_reward *= reward.sum() / num_ped
    
    global_reward_wo_coll = no_coll_reward.sum() / num_ped
    g_coll = collision.sum() / num_ped
    return reward, n_reward, g_reward, global_reward_wo_coll, g_coll

def make_env(args, env_name, seed, worker_id):
    
    home_path = args.home_path
    env_path = home_path + args.env_path + env_name + ".x86_64"
    
    channel = EngineConfigurationChannel()
    unity_env = UE(file_name = env_path, seed=seed, side_channels=[channel], no_graphics=not args.eval_rendering, worker_id=worker_id)
    
    env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True)
    channel.set_configuration_parameters(time_scale = 8.0)
    
    return env

def check_success_rate(obs, l, w):
    
    x = obs[:,0] * (l//2)
    z = obs[:,1] * (w//2)
    goal_x = obs[:,3] * (l//2)
    goal_z = obs[:,4] * (w//2)
    distance_to_goal = np.sqrt((x-goal_x)**2 + (z-goal_z)**2)
    success = distance_to_goal < 0.5

    return np.mean(success), success

def shortest_distance(obs, l, w, num_ped):
    l2_distance = np.zeros(num_ped)
    
    x = obs[:,0] * (l//2)
    z = obs[:,1] * (w//2)
    
    goal_x = obs[:,3] * (l//2)
    goal_z = obs[:,4] * (w//2)

    l2_distance = np.sqrt((x - goal_x)**2 + (z - goal_z) ** 2) 
    return l2_distance
