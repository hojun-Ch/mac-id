import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

from mlagents_envs.environment import UnityEnvironment as UE
from gym_unity.envs import UnityToGymWrapper

from gym.spaces import Dict, Box
import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal
from models.model import Encoder, PolicyNetwork, ValueNetwork_mf, lcfNetwork
from buffers.buffers import MeanFieldRolloutBuffer


class MFPO():
    
    def __init__(self, args):
        
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.n_envs = args.num_ped
        self.learning_rate = args.learning_rate
        self.gae_lambda = args.gae_lambda
        self.gamma = args.gamma
        self.buffer_size = args.buffer_size
        self.ppo_epoch = args.ppo_epoch
        self.batch_size = args.batch_size
        self.n_updates = 0
        self.clip_range = args.ppo_clip_range
        self.normalize_advantage = args.normalize_advantages
        self.ent_coef = args.ent_coef
        self.vf_coef = args.vf_coef
        self.max_grad_norm = args.max_grad_norm
        
        # rollout buffer
        self.rollout_buffer = MeanFieldRolloutBuffer(self.buffer_size, self.obs_dim, self.action_dim, device=args.device, gae_lambda=self.gae_lambda, gamma=self.gamma, n_envs=self.n_envs)
        
        # networks
        self.policy_network = PolicyNetwork(args)
        self.i_value_network = ValueNetwork_mf(args)

        self.device = torch.device(args.device)
        self.policy_network.to(self.device)
        self.i_value_network.to(self.device)

                
        # optimizers
        entire_parameters = list(self.policy_network.parameters()) + list(self.i_value_network.parameters())
  
        self.optimizer = torch.optim.Adam(entire_parameters, lr=self.learning_rate)

        
    def evaluate_actions(self, obs, nearby_obs, actions):

        obs = obs.to(self.device)
            
        i_value = self.i_value_network(obs, nearby_obs)

        
        log_prob, entropy = self.policy_network.evaluate_action(obs, actions)
            
        return i_value, log_prob, entropy
    
    def evaluate_actions_with_old(self, obs, actions):
        
        obs = obs.to(self.device)
        log_prob, __ = self.old_policy_network.evaluate_action(obs, actions)
        
        return log_prob
        
        
    def act(self, obs, nearby_obs):
        """
        """
        with torch.no_grad():
            obs = obs.to(self.device)
            nearby_obs = nearby_obs.to(self.device)
            
            action, log_prob = self.policy_network(obs)
            
            value = self.i_value_network(obs, nearby_obs)
        
        return action.clone().cpu().numpy(), log_prob, value
    
    def get_values(self, obs, nearby_obs):
        """
        """
        with torch.no_grad():
            obs = obs.to(self.device)
            nearby_obs = nearby_obs.to(self.device)
                        
            value = self.i_value_network(obs, nearby_obs)
        
        return value

    def update_policy(self):
        clip_range = self.clip_range
        self.old_policy_network = copy.deepcopy(self.policy_network)
        self.old_optimizer = torch.optim.Adam(list(self.old_policy_network.parameters()))
        
        pg_losses = []
        clip_fractions = []
        entropy_losses = []
        value_losses = []

        for epoch in range(self.ppo_epoch):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                values, log_prob, entropy = self.evaluate_actions(rollout_data.observations, rollout_data.nearby_observations ,actions)
                values = values.flatten()

                # Normalize advantage
                i_advantages = rollout_data.i_advantages
                
                if self.normalize_advantage:
                    i_advantages = (i_advantages - i_advantages.mean()) / (i_advantages.std() + 1e-8)
                    
                advantages =  i_advantages
                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                values_pred = values
 
                    
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.i_returns, values_pred)
                value_losses.append(value_loss.item())
                

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()

                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                                
        self.n_updates += self.ppo_epoch
        return pg_losses, clip_fractions, value_losses
    
                
    def train_mode(self):
        self.policy_network.train()
        self.i_value_network.train()

    def eval_mode(self):
        self.policy_network.eval()
        self.i_value_network.eval()
        
    def save_model(self, path):
        torch.save({
            'policy_state_dict':self.policy_network.state_dict(),
            'i_value_state_dict':self.i_value_network.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
        }, path)
        
    def load_ckpt(self, model_path):
        
        ckpt = torch.load(model_path)
        self.policy_network.load_state_dict(ckpt['policy_state_dict'])
        self.i_value_network.load_state_dict(ckpt['i_value_state_dict'])

        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return
        