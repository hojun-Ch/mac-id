import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='pedsim')
parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help='random seed'
)
parser.add_argument(
    '--algo',
    type=str,
    default='copo',
    help='policy algo'
)

parser.add_argument(
    '--name',
    type=str,
    default='demo',
    help='name of each trial'
)

parser.add_argument(
    '--env_name',
    type=str,
    default='bottleneck',
    help='environment name'
)

parser.add_argument(
    '--worker',
    type=int,
    default=0,
    help='worker id'
)
# env path
parser.add_argument(
    '--home_path',
    type=str,
    default='../',
    help='path to home')

parser.add_argument(
    '--env_path',
    type=str,
    default="pedsim_lidar_env/",
    help='path from home to env folder')

# archive
parser.add_argument(
    '--archive_frequency',
    type=int,
    default=10,
    help='Model save frequency'
)
# for COPO training
parser.add_argument(
    '--train_lcf',
    type=str2bool,
    default=True,
    help='train lcf or not'
)
parser.add_argument(
    '--rendering',
    type=str2bool,
    default=False,
    help='render while training')
parser.add_argument(
    '--state_dim',
    type=int,
    default=4805,
    help='dim of state')
parser.add_argument(
    '--obs_dim',
    type=int,
    default=38,
    help='dim of observation'
)
parser.add_argument(
    '--dummy_index',
    type=int,
    default=52,
    help='offsets for dummy obs from agent lidar sensor'
)

parser.add_argument(
    '--action_dim',
    type=int,
    default=2,
    help='dim of action')

parser.add_argument(
    '--num_ped',
    type=int,
    default=51,
    help='# of pedestrians')

parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-3,
    help='learning rate of policy')

parser.add_argument(
    '--gae_lambda',
    type=float,
    default=0.95,
    help='lambda for calculating Generalized Advantage Estimation')

parser.add_argument(
    '--gamma',
    type=float,
    default=0.999,
    help='discounting factor')

parser.add_argument(
    '--buffer_size',
    type=int,
    default=1000,
    help='size of buffers (must same with rollout_length)')

parser.add_argument(
    '--ppo_epoch',
    type=int,
    default=5,
    help='ppo epoch for one training loop')

parser.add_argument(
    '--batch_size',
    type=int,
    default=2048,
    help='batch size for training')
parser.add_argument(
    '--ppo_clip_range',
    type=float,
    default=0.2,
    help='batch size for training')
parser.add_argument(
    '--normalize_advantages',
    type=str2bool,
    default=True,
    help='normalize advantage for stable training')
parser.add_argument(
    '--lcf_learning_rate',
    type=float,
    default=1e-4,
    help='learning rate of local coordinate factor')
parser.add_argument(
    '--lcf_epochs',
    type=int,
    default=5,
    help='lcf epoch for one training loop')
parser.add_argument(
    '--ent_coef',
    type=float,
    default=0.01,
    help='ent coef for entropy loss')
parser.add_argument(
    '--vf_coef',
    type=float,
    default=1.0,
    help='vf coef for value loss')
parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=0.5,
    help='max grad norm for gradient clipping')
parser.add_argument(
    '--device',
    type=str,
    default='cuda',
    help='learning device(gpu or cpu)'
)
parser.add_argument(
    '--rollout_length',
    type=int,
    default=1000
)
parser.add_argument(
    '--max_step',
    type=int,
    default=1000000,
    help='total env step'
)

# for CCPO artchitecture

parser.add_argument(
    '--conditioned_value',
    type=bool,
    default=False,
    help='use a value function conditioned on local coordinate factor'
)

# for network architecture

parser.add_argument(
    '--dropout',
    type=float,
    default=0.0,
    help='dropout prob')
parser.add_argument(
    '--encoder_num_hidden',
    type=int,
    default=64,
    help='dim of state encoding'
)
parser.add_argument(
    '--policy_num_hidden',
    type=int,
    default=64,
    help='hidden layer dim inside the actor&critic network'
)

# for environment
parser.add_argument(
    '--sparse_reward',
    type=str2bool,
    default=False,
    help='get reward if agent reach the goal only'
)
parser.add_argument(
    '--map_length',
    type=int,
    default=50,
    help='length of the entire map'
)

parser.add_argument(
    '--map_width',
    type=int,
    default=50,
    help='width of the entire map'
)

parser.add_argument(
    '--neighbor_distance',
    type=int,
    default=5,
    help='radius to define neighbor'
)

parser.add_argument(
    '--coll_penalty',
    type=float,
    default=1.0,
    help='collision penalty'
)

parser.add_argument(
    '--smooth_cost',
    type=str2bool,
    default=False,
    help='use smoothed collision penalty'
)
# for evaluation
parser.add_argument(
    '--eval_frequency',
    type=int,
    default=50,
    help='evaluation frequency'
)

parser.add_argument(
    '--eval_rendering',
    type=str2bool,
    default=False,
    help='render evaluation'
)

parser.add_argument(
    '--model_path',
    type=str,
    default='./ckpts/',
    help='path to saved model'
)

# SAC parameters
parser.add_argument(
    '--initial_temperature',
    type=float,
    default=0.1,
    help='initial temperature'
)
parser.add_argument(
    '--learnable_temperature',
    type=str2bool,
    default=True,
    help='learn entropy coeff'
)
parser.add_argument(
    '--critic_tau',
    type=float,
    default=0.005,
    help='critic ema'
)
parser.add_argument(
    '--learning_starts',
    type=int,
    default=5000,
    help='start learning after this value env step'
)
parser.add_argument(
    '--critic_target_update_freq',
    type=int,
    default=2,
)