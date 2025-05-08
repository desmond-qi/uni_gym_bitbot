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

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  self.commands[:, :3] * self.commands_scale,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)

        self.privileged_obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                             self.base_ang_vel * self.obs_scales.ang_vel,
                                             self.projected_gravity,
                                             self.commands[:, :3] * self.commands_scale,
                                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                             self.dof_vel * self.obs_scales.dof_vel,
                                             self.actions
                                             ), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _init_buffers(self):
        super()._init_buffers()
        rigid_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_state = gymtorch.wrap_tensor(rigid_state)
        self.rigid_state_ = self.rigid_state.view(self.num_envs, self.num_bodies, 13)
        self.rigid_rotation = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self.rigid_position = self.rigid_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]

        self.feet_num = len(self.feet_indices)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

        return super()._post_physics_step_callback()


    def _resample_commands(self, env_ids):
        if self.isTrain:
            self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            if self.cfg.commands.heading_command:
                self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            else:
                self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            # for i in range(4):
            #     self.commands[env_ids, i] = torch_rand_float(0, 0, (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 0] = 0.0
            self.commands[env_ids, 1] = 0.0
            self.commands[env_ids, 2] = 0.0
            self.commands[env_ids, 3] = 0.

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    if self.cfg.domain_rand.randomize_PD:
                        self.p_gains[i] = self.cfg.control.stiffness[dof_name] * np.random.uniform(self.cfg.domain_rand.Kp_ratio_bias_range[0], self.cfg.domain_rand.Kp_ratio_bias_range[1])
                        self.d_gains[i] = self.cfg.control.damping[dof_name] * np.random.uniform(self.cfg.domain_rand.Kd_ratio_bias_range[0], self.cfg.domain_rand.Kd_ratio_bias_range[1])
                    else:
                        self.p_gains[i] = self.cfg.control.damping[dof_name]
                        self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")

    def _reward_double_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        double_contact = torch.sum(1.*contacts, dim=1)==2
        return 1.*double_contact
    
    def _reward_double_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] < 0.1
        double_no_contact = torch.sum(1.*contacts, dim=1)==2
        return 1.*double_no_contact

    # def _reward_contact(self):
    #     res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    #     for i in range(self.feet_num):
    #         contact = self.contact_forces[:, self.feet_indices[i], 2] > 0.1
    #         res += torch.sum(1.*contact)==1
    #     return 1.*res

    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res

    def _reward_foot_posture(self):
        trunk_euler = get_euler_xyz(self.rigid_rotation[:, 0, :])
        leftFoot_roll = trunk_euler[0] + self.dof_pos[:, 1] + self.dof_pos[:, 5]
        rightFoot_roll = trunk_euler[0] + self.dof_pos[:, 8] + self.dof_pos[:, 12]
        return torch.abs(leftFoot_roll) + torch.abs(rightFoot_roll)
    
    # def _reward_footAngVel(self):
    #     return torch.abs(self.dof_vel[:, 3]) + torch.abs(self.dof_vel[:, 8])
    
    def _reward_arm_symmetry(self):
        return torch.abs(self.dof_pos[:, 6] - self.dof_pos[:, 13])
    
    # def _reward_armPosition(self):
    #     return torch.abs(self.dof_pos[:, 10]) + torch.abs(self.dof_pos[:, 11])
    
    def _reward_hip_symmetry(self):
        return torch.abs(self.dof_pos[:, 2] - self.dof_pos[:, 9]) + torch.abs(self.dof_pos[:, 1] - self.dof_pos[:, 8]) + torch.abs(self.dof_pos[:, 0] - self.dof_pos[:, 7])
    
    def _reward_arm_velocity(self):
        return torch.abs(self.dof_vel[:, 6]) + torch.abs(self.dof_vel[:, 13])
    
    def _reward_posture_roll(self):
        trunk_euler = get_euler_xyz(self.rigid_rotation[:, 0, :])
        return torch.abs(trunk_euler[0])

