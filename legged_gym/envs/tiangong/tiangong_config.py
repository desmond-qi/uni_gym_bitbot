from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
from scipy.spatial.transform import Rotation as Rot

D2R = np.pi / 180.0

class TiangongCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_observations = 51
        num_privileged_obs = 54
        num_actions = 14

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class commands(LeggedRobotCfg.commands):
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-2.0, 2.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.1]  # Initial position x, y, z [m]
        # initPos_euler = [0.0, -0.0 * D2R, 0.0]
        # trans = Rot.from_euler('xyz', initPos_euler)
        # initPos_quate = trans.as_quat()
        # rot = [initPos_quate[0], initPos_quate[1], initPos_quate[2], initPos_quate[3]]
        default_joint_angles = {  # Target angles [rad] when action = 0.0
            'lhip_roll_joint': 0.0 * D2R,
            'lhip_yaw_joint': 0.0 * D2R,
            'lhip_pitch_joint': -20.0 * D2R, #'hip_y_left': 70.0 * D2R,
            'lknee_pitch_joint': 40.0 * D2R, #'knee_left': -100.0 * D2R,
            'lankle_pitch_joint': -20.0 * D2R, #'ankle_y_left': 46.0 * D2R,
            'lankle_roll_joint': 0.0 * D2R,
            'lshoulder_pitch_joint': 0.0 * D2R,

            'rhip_roll_joint': 0.0 * D2R,
            'rhip_yaw_joint': 0.0 * D2R,
            'rhip_pitch_joint': -20.0 * D2R, #'hip_y_left': 70.0 * D2R,
            'rknee_pitch_joint': 40.0 * D2R, #'knee_left': -100.0 * D2R,
            'rankle_pitch_joint': -20.0 * D2R, #'ankle_y_left': 46.0 * D2R,
            'rankle_roll_joint': 0.0 * D2R,
            'rshoulder_pitch_joint': 0.0 * D2R,
        }

    class control(LeggedRobotCfg.control):
        stiffness = {'hip_roll_joint': 150,
                     'hip_yaw_joint': 150,
                     'hip_pitch_joint': 150,
                     'knee_pitch_joint': 200,
                     'ankle_pitch_joint': 40,
                     'ankle_roll_joint': 40,
                     'shoulder_pitch_joint': 50,
                     }  # [N*m/rad]
        damping = {  'hip_roll_joint': 2,
                     'hip_yaw_joint': 2,
                     'hip_pitch_joint': 2,
                     'knee_pitch_joint': 4,
                     'ankle_pitch_joint': 2,
                     'ankle_roll_joint': 2,
                     'shoulder_pitch_joint': 1,
                     } # [N*m*s/rad]
        action_scale = 0.25  # Scale for actions
        decimation = 4  # Number of control action updates per policy update

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/tiangong/urdf/tiangong_chz.urdf'
        name = "tiangong"
        foot_name = 'ankle_roll'
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 0  # 1 to disable, 0 to enable (bitwise filter)

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 800.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 1.0
            tracking_lin_vel = 5.0
            torques = -5.e-6
            dof_acc = -2.e-6
            lin_vel_z = -2.e-3
            feet_air_time = 1.
            dof_pos_limits = -1.
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            #qhx
            # double_fly = 0.0
            # double_no_fly = 0.5
            contact = 0.5
            hip_symmetry = -1.e-1
            # foot_posture = .0
            # footAngVel = -1.e-5
            arm_symmetry = -1.e-1
            # armPosition = -1.e-5
            arm_velocity = -2.e-2
            posture_roll = -2.e-1
    
    # class viewer(LeggedRobotCfg.viewer):
        # pos = [3, 0, 1]

class TiangongCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [64, 32]
        critic_hidden_dims = [64, 32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'tiangong_training'
        experiment_name = 'tiangong_experiment'
        max_iterations = 3000

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.02
        learning_rate = 3e-4
        gamma = 0.99
