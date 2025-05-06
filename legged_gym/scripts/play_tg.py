import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from pynput import keyboard


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.vel_debug:
        env_cfg.env.num_envs = 16
        env_cfg.env.velocity_debug = True
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 4

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    print("****************")
    print(env.dof_names)
    robot_idx = 0
    # init logger
    if args.show_log:
        dof_names = env.dof_names
        dof_pos_limits = env.dof_pos_limits
        logger = Logger(env.dt, dof_names)
        robot_idx = 0
        stop_record_log = 150
        action_scale = env_cfg.control.action_scale

        # get the dir to loaded run and model: 
        log_root_plt = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        runs = os.listdir(log_root_plt)
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        loaded_run = -1
        model_plt = -1
        if train_cfg.runner.load_run == -1:
            loaded_run = runs[-1]
        else:
            loaded_run = train_cfg.runner.load_run
        # get the last checkpoint in dir last_run
        if train_cfg.runner.checkpoint == -1:
            models_plt = [file for file in os.listdir(os.path.join(log_root_plt, loaded_run)) if 'model' in file]
            models_plt.sort(key=lambda m: '{0:0>15}'.format(m))
            model_plt = models_plt[-1].split('.')[0]
        else:
            model_plt  = f'model_{train_cfg.runner.checkpoint}'
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # chz
    command_ranges = env.cfg.commands.ranges
    dphase_bounds = env.cfg.commands.ranges.dphase
    keyboard_commands = torch.zeros(5, device=env.device)
    keyboard_commands[4] = (dphase_bounds[0] + dphase_bounds[1]) / 2
    reset_commands = torch.zeros(env_cfg.env.num_envs, device=env.device)
    def on_press(key):
        print(key)
        try:
            if key.char == 'w':
                keyboard_commands[0] += 0.33
            elif key.char == 's':
                keyboard_commands[0] -= 0.33
            elif key.char == 'a':
                keyboard_commands[1] += 0.33
            elif key.char == 'd':
                keyboard_commands[1] -= 0.33
            elif key.char == 'j':
                keyboard_commands[2] += 0.33
            elif key.char == 'l':
                keyboard_commands[2] -= 0.33
            elif key.char == 'k':
                keyboard_commands[2] = 0
            elif key.char == 'q':
                keyboard_commands[:3] = 0
                keyboard_commands[4] = (dphase_bounds[0] + dphase_bounds[1]) / 2
            elif key.char == 'u':
                keyboard_commands[4] += 0.33
            elif key.char == 'o':
                keyboard_commands[4] -= 0.33
            elif key.char == 'i':
                keyboard_commands[4] = (dphase_bounds[0] + dphase_bounds[1]) / 2
            elif key.char == 'r':
                reset_commands.fill_(1)
        except AttributeError:
            pass
        keyboard_commands[:2].clip_(min=command_ranges.lin_vel_x[0], max=command_ranges.lin_vel_x[1])
        keyboard_commands[2].clip_(min=command_ranges.ang_vel_yaw[0], max=command_ranges.ang_vel_yaw[1])
        keyboard_commands[3].clip_(min=command_ranges.heading[0], max=command_ranges.heading[1])
        keyboard_commands[4].clip_(min=command_ranges.dphase[0], max=command_ranges.dphase[1])
        
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


    for i in range(10*int(env.max_episode_length)):
        env.commands[:, :3] = keyboard_commands[:3]
        env.commands[:, 4:5] = keyboard_commands[4:5]
        env.commands[:, 5] = 0.35
        env.commands[:, 6] = 0.50
        actions = policy(obs.detach())
        obs, pobs, rews, dones, infos = env.step(actions.detach())

        if args.show_log:
            if i < stop_record_log:
                logger.log_joint_state(robot_index=robot_idx, dof_pos=env.dof_pos, action=actions, ref_pos=None)
            elif i == stop_record_log:
                logger.plot_states(train_cfg.runner.experiment_name, loaded_run, model_plt , dof_pos_limits, log_root_plt, action_scale)    
        
        # chz
        # print obs and actions to file
        logfilepath = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'data')
        obs_data = ','.join(f"{val:.10f}" for val in obs[robot_idx].cpu().detach().numpy())
        privobs_data = ','.join(f"{val:.10f}" for val in pobs[robot_idx].cpu().detach().numpy())
        actions_data = ','.join(f"{val:.10f}" for val in actions[robot_idx].cpu().detach().numpy())

        if i == 0:
            num_dofs = env_cfg.env.num_dofs
            with open(logfilepath + '/chzdata.csv', 'w') as f:
                f.write('avel_roll,avel_pitch,avel_yaw,grav_x,grav_y,grav_z,command_vx,command_vy,command_dyaw,command_dphase,')
                for i in range(int(num_dofs/2)):
                    f.write(f'pos_l{i+1},')
                for i in range(int(num_dofs/2)):
                    f.write(f'pos_r{i+1},')
                for i in range(int(num_dofs/2)):
                    f.write(f'vel_l{i+1},')
                for i in range(int(num_dofs/2)):
                    f.write(f'vel_r{i+1},')
                for i in range(int(num_dofs/2)):
                    f.write(f'action_l{i+1},')
                for i in range(int(num_dofs/2)):
                    f.write(f'action_r{i+1},')
                f.write('action_dphase,sin_phase,cos_phase,lvel_x,lvel_y,lvel_z,')
                f.write('\n')
                        
        with open(logfilepath + '/chzdata.csv', 'a') as f:
            # f.write(obs_data + ',' + actions_data + ',' + str(symmetry_loss_numpy[0]) + ',')
            f.write(privobs_data + ',')
            f.write('\n')
    
    # chz
    listener.join()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)