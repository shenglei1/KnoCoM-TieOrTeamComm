import random
from collections import namedtuple,deque
import torch
import numpy as np


class Loggeer:

    def __init__(self, exp_config, swanlab):

        self.algo = exp_config['agent']
        self.swanlab = swanlab



    def log_agent(self, epoch,total_num_episodes,epoch_time,total_num_steps,log):
        self.swanlab.log({"epoch": epoch,
                   'episode': total_num_episodes,
                   'epoch_time': epoch_time,
                   'total_steps': total_num_steps,
                   'episode_return': np.mean(log['episode_return']),
                   "episode_steps": np.mean(log['episode_steps']),
                   'action_loss': log['action_loss'],
                   'value_loss': log['value_loss'],
                   'total_loss': log['total_loss'],
                   })


    def log_tj(self,**kwargs):
        self.swanlab.log({"epoch": kwargs['epoch'],
                   'episode': kwargs['episode'],
                   })






