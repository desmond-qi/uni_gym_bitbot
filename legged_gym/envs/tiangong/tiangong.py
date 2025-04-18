from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs.base.legged_robot import LeggedRobot

from scipy.spatial.transform import Rotation as R
from isaacgym.torch_utils import get_euler_xyz

class Tiangong(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, isTrain):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.isTrain = isTrain

    def _init_buffers(self):
        super()._init_buffers()
        rigid_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_state = gymtorch.wrap_tensor(rigid_state)
        self.rigid_state_ = self.rigid_state.view(self.num_envs, self.num_bodies, 13)
        self.rigid_rotation = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self.rigid_position = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]

    def _resample_commands(self, env_ids):
        if self.isTrain:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 0] = 1.5
            self.commands[env_ids, 1] = 0.0
            self.commands[env_ids, 2] = 0.0
            self.commands[env_ids, 3] = 0.

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _reward_double_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        double_contact = torch.sum(1.*contacts, dim=1)==2
        return 1.*double_contact
    
    def _reward_double_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] < 0.1
        double_no_contact = torch.sum(1.*contacts, dim=1)==2
        return 1.*double_no_contact
    
    def _reward_footPosture(self):
        trunk_euler = get_euler_xyz(self.rigid_rotation[:, 0, :])
        leftFoot_roll = trunk_euler[0] + self.dof_pos[:, 1] + self.dof_pos[:, 5]
        rightFoot_roll = trunk_euler[0] + self.dof_pos[:, 8] + self.dof_pos[:, 12]
        return torch.abs(leftFoot_roll) + torch.abs(rightFoot_roll)
    
    # def _reward_footAngVel(self):
    #     return torch.abs(self.dof_vel[:, 3]) + torch.abs(self.dof_vel[:, 8])
    
    # def _reward_armSymmetry(self):
    #     return torch.abs(self.dof_pos[:, 10] - self.dof_pos[:, 11])
    
    # def _reward_armPosition(self):
    #     return torch.abs(self.dof_pos[:, 10]) + torch.abs(self.dof_pos[:, 11])
    
    def _reward_hipSymmetry(self):
        return torch.abs(self.dof_pos[:, 2] - self.dof_pos[:, 9])

