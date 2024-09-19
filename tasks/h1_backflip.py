# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from isaacgym import torch_utils
from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.tasks.base.vec_task import VecTask

import numpy as np
import torch
import os

class Env(VecTask):
    def __init__(self, 
                 cfg: dict, 
                 rl_device: str, 
                 sim_device: str, 
                 graphics_device_id: int, 
                 headless: bool, 
                 virtual_screen_capture: bool, 
                 force_render: bool,
                 ):

        """
        raw_obs: 
            base orn (3), joint pos (19), joint vel (19), prev action (19), central phase (4), command (3)
        observation: raw_obs * history_len
        state: 
            linear vel (3), angular vel (3), com height (1), hand/foot contact (4), stage (5)
        """
        self.cfg = cfg
        self.raw_obs_dim = 3 + 19*3 + 4 + 3
        self.history_len = self.cfg["env"]["history_len"]
        self.cfg['env']['numObservations'] = self.raw_obs_dim * self.history_len
        self.cfg['env']['numStates'] = 3 + 3 + 1 + 4 + 5 
        self.sim_dt = self.cfg["sim"]["sim_dt"]
        self.control_dt = self.cfg["sim"]["con_dt"]
        self.cfg["env"]["controlFrequencyInv"] = int(self.control_dt/self.sim_dt + 0.5)
        self.cfg['sim']['dt'] = self.sim_dt
        self.cfg['physics_engine'] = self.cfg['sim']['physics_engine']
        self.cfg['env']['enableCameraSensors'] = self.cfg['env']['enable_camera_sensors']
        self.cfg['env']['numEnvs'] = self.cfg['env']['num_envs']
        self.cfg['env']['numActions'] = 19

        """
        In the parent's __init__, generate the following member variables:
            - self.device_type: {cuda, gpu, cpu}
            - self.device_id: {0, 1, ...}
            - self.device: {cuda:0, cuda:1, ..., cpu}
            - self.rl_device
            - self.headless
            - self.graphics_device_id
            - self.num_environments: config['env']['numEnvs']
            - self.num_observations: config['env']['numObservations']
            - self.num_actions: config['env']['numActions']
            - self.control_freq_inv: {1, config['env']['controlFrequencyInv']}
            - self.obs_space, self.act_space
            - self.clip_obs, self.clip_actions
            - self.virtual_screen_capture, self.force_render
            - self.sim_params: self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
            - self.physics_engine: {gymapi.SIM_PHYSX, gymapi.SIM_FLEX}
        Then, called:
            - self.create_sim()
            - self.set_viewer()
            - self.allocate_buffers()
        """
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, 
                         graphics_device_id=graphics_device_id, headless=headless, 
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.max_episode_length_s = self.cfg["env"]["learn"]["episode_length_s"]
        self.max_episode_length = int(self.max_episode_length_s/self.control_dt + 0.5)
        self.reward_names = self.cfg["env"]["reward_names"]
        self.cost_names = self.cfg["env"]["cost_names"]
        self.stage_names = self.cfg["env"]["stage_names"]
        self.num_rewards = len(self.reward_names)
        self.num_costs = len(self.cost_names)
        self.num_stages = len(self.stage_names)
        self.action_smooth_weight = self.cfg["env"]["control"]["action_smooth_weight"]
        self.action_scale = self.cfg["env"]["control"]["action_scale"]
        self.gait_freq = torch_utils.to_torch(
            self.cfg["env"]["learn"]["gait_frequency"], 
            dtype=torch.float32, device=self.device, requires_grad=False)

        # for buffer
        self.rew_buf = torch.zeros((self.num_envs, self.num_rewards),
                                    dtype=torch.float32, device=self.device)
        self.cost_buf = torch.zeros((self.num_envs, self.num_costs), 
                                    dtype=torch.float32, device=self.device)
        self.stage_buf = torch.zeros((self.num_envs, self.num_stages),
                                    dtype=torch.float32, device=self.device)
        self.fail_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_half_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.is_one_turn_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.start_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.cmd_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.land_time_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # get gym state tensors
        """
        It can be considered as the following variables get GPU pointers.
        After refresh state tensors, value of the following variables are updated automatically.
        """
        raw_actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        raw_dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        raw_dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        raw_net_contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        raw_rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # update state and force tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(
                            raw_actor_root_state_tensor).view(self.num_envs, 13)
        self.dof_states = gymtorch.wrap_tensor(
                            raw_dof_state_tensor).view(self.num_envs, self.num_dofs, 2)
        self.dof_torques = gymtorch.wrap_tensor(
                            raw_dof_force_tensor).view(self.num_envs, self.num_dofs)
        self.contact_forces = gymtorch.wrap_tensor(
                            raw_net_contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        self.rigid_body_states = gymtorch.wrap_tensor(
                            raw_rigid_body_state_tensor).view(self.num_envs, self.num_bodies, 13)
        self.base_positions = self.root_states[:, :3]
        self.base_quaternions = self.root_states[:, 3:7]
        self.base_lin_vels = self.root_states[:, 7:10]
        self.base_ang_vels = self.root_states[:, 10:13]
        self.dof_positions = self.dof_states[..., 0]
        self.dof_velocities = self.dof_states[..., 1]

        # setting viewer
        if self.viewer != None:
            p = self.base_positions[0]
            cam_pos = gymapi.Vec3(p[0] + 5.0, p[1] + 5.0, p[2] + 3.0)
            cam_target = gymapi.Vec3(*p)
            self.gym.viewer_camera_look_at(
                self.viewer, self.env_handles[0], cam_pos, cam_target)

        # get default base frame pose
        base_init_state = []
        base_init_state += self.cfg["env"]["init_base_pose"]["pos"]
        base_init_state += self.cfg["env"]["init_base_pose"]["quat"]
        base_init_state += self.cfg["env"]["init_base_pose"]["lin_vel"]
        base_init_state += self.cfg["env"]["init_base_pose"]["ang_vel"]
        self.base_init_state = torch_utils.to_torch(
            base_init_state, dtype=torch.float32, device=self.device, requires_grad=False)

        # get default joint (DoF) position
        self.named_default_joint_positions = self.cfg["env"]["default_joint_positions"]
        self.named_sit_joint_positions = self.cfg["env"]["sit_joint_positions"]
        self.default_dof_positions = torch.zeros_like(
            self.dof_positions, dtype=torch.float32, device=self.device, requires_grad=False)
        self.sit_dof_positions = torch.zeros_like(
            self.dof_positions, dtype=torch.float32, device=self.device, requires_grad=False)
        for i in range(self.num_actions):
            name = self.dof_names[i]
            self.default_dof_positions[:, i] = self.named_default_joint_positions[name]
            self.sit_dof_positions[:, i] = self.named_sit_joint_positions[name]

        # for inner variables
        self.world_x = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.world_y = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.world_z = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.robot_up = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.world_x[:, 0] = 1.0
        self.world_y[:, 1] = 1.0
        self.world_z[:, 2] = 1.0
        self.robot_up[:, 2] = 0.5
        self.joint_targets = torch.zeros(
            (self.num_envs, self.num_dofs), 
            dtype=torch.float32, device=self.device, requires_grad=False)
        self.prev_actions = torch.zeros(
            (self.num_envs, self.num_dofs), 
            dtype=torch.float32, device=self.device, requires_grad=False)

        # for dof limits
        self.dof_pos_lower_limits = torch_utils.to_torch(
            self.cfg["env"]["learn"]["joint_pos_limit"]["lower"], dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_pos_lower_limits = torch.maximum(self.dof_pos_lower_limits, self.default_dof_pos_lower_limits)
        self.dof_pos_upper_limits = torch_utils.to_torch(
            self.cfg["env"]["learn"]["joint_pos_limit"]["upper"], dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_pos_upper_limits = torch.minimum(self.dof_pos_upper_limits, self.default_dof_pos_upper_limits)
        self.dof_vel_upper_limits = torch_utils.to_torch(
            self.cfg["env"]["learn"]["joint_vel_upper"], dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_vel_upper_limits = torch.minimum(self.dof_vel_upper_limits, self.default_dof_vel_upper_limits)
        self.dof_torques_upper_limits = torch_utils.to_torch(
            self.cfg["env"]["learn"]["joint_torque_upper"], dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_torques_upper_limits = torch.minimum(self.dof_torques_upper_limits, self.default_dof_torques_upper_limits)

        # for observation and action symmetric matrix
        self.joint_sym_mat = torch.zeros((self.num_dofs, self.num_dofs), device=self.device, dtype=torch.float32, requires_grad=False)
        self.joint_sym_mat[0:5, 5:10] = torch.eye(5, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[0, 5] *= -1.0
        self.joint_sym_mat[1, 6] *= -1.0
        self.joint_sym_mat[5:10, 0:5] = torch.eye(5, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[5, 0] *= -1.0
        self.joint_sym_mat[6, 1] *= -1.0
        self.joint_sym_mat[10, 10] = -1.0
        self.joint_sym_mat[11:15, 15:19] = torch.eye(4, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[12, 16] *= -1.0
        self.joint_sym_mat[13, 17] *= -1.0
        self.joint_sym_mat[15:19, 11:15] = torch.eye(4, device=self.device, dtype=torch.float32)
        self.joint_sym_mat[16, 12] *= -1.0
        self.joint_sym_mat[17, 13] *= -1.0
        self.obs_sym_mat = torch.zeros((self.num_obs, self.num_obs), device=self.device, dtype=torch.float32, requires_grad=False)
        raw_obs_sym_mat = torch.eye(self.raw_obs_dim, device=self.device, dtype=torch.float32, requires_grad=False)
        raw_obs_sym_mat[1, 1] = -1.0
        for i in range(3):
            raw_obs_sym_mat[(3+self.num_dofs*(i)):(3+self.num_dofs*(i+1)), (3+self.num_dofs*(i)):(3+self.num_dofs*(i+1))] = self.joint_sym_mat.clone()
        raw_obs_sym_mat[3+3*self.num_dofs:7+3*self.num_dofs, 3+3*self.num_dofs:7+3*self.num_dofs] = 0.0
        raw_obs_sym_mat[3+3*self.num_dofs:5+3*self.num_dofs, 5+3*self.num_dofs:7+3*self.num_dofs] = torch.eye(2, device=self.device, dtype=torch.float32)
        raw_obs_sym_mat[5+3*self.num_dofs:7+3*self.num_dofs, 3+3*self.num_dofs:5+3*self.num_dofs] = torch.eye(2, device=self.device, dtype=torch.float32)
        raw_obs_sym_mat[7+3*self.num_dofs:, 7+3*self.num_dofs:] = torch.eye(3, device=self.device, dtype=torch.float32)        
        for i in range(self.history_len):
            self.obs_sym_mat[(self.raw_obs_dim*i):(self.raw_obs_dim*(i+1)), (self.raw_obs_dim*i):(self.raw_obs_dim*(i+1))] = raw_obs_sym_mat.clone()
        self.state_sym_mat = torch.eye(self.num_states - self.num_stages, device=self.device, dtype=torch.float32, requires_grad=False)
        self.state_sym_mat[1, 1] = -1.0
        self.state_sym_mat[3, 3] = -1.0
        self.state_sym_mat[5, 5] = -1.0
        self.state_sym_mat[7:11, 7:11] = 0
        self.state_sym_mat[7, 8] = 1.0
        self.state_sym_mat[8, 7] = 1.0
        self.state_sym_mat[9, 10] = 1.0
        self.state_sym_mat[10, 9] = 1.0


    def create_sim(self):
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # make terrain
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

        # set asset location
        asset_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/assets'
        asset_file = self.cfg["env"]["urdf_asset"]["file"]
        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        # create actor (or robot)'s asset option
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.collapse_fixed_joints = True
        asset_options.fix_base_link = self.cfg["env"]["urdf_asset"]["fix_base_link"]
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = self.cfg['env']['urdf_asset']['flip_visual_attachments']
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 1000.0
        asset_options.max_linear_velocity = 1000.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01

        # load asset
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset) # 19
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset) # 20

        # set friction and restitution
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(robot_asset)
        for s in range(len(rigid_shape_prop)):
            rigid_shape_prop[s].friction = 1.0
            rigid_shape_prop[s].restitution = 0.0
        self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_prop)

        # get an set dof property
        self.link_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for i, joint_name in enumerate(self.dof_names):
            for dof_name in self.cfg["env"]["control"]["stiffness"]:
                if dof_name in joint_name:
                    dof_props['stiffness'][i] = self.cfg["env"]["control"]["stiffness"][dof_name]
                    dof_props['damping'][i] = self.cfg["env"]["control"]["damping"][dof_name]
        self.default_dof_pos_lower_limits = torch_utils.to_torch(
            dof_props['lower'], dtype=torch.float32, device=self.device, requires_grad=False)
        self.default_dof_pos_upper_limits = torch_utils.to_torch(
            dof_props['upper'], dtype=torch.float32, device=self.device, requires_grad=False)
        self.default_dof_vel_upper_limits = torch_utils.to_torch(
            dof_props['velocity'], dtype=torch.float32, device=self.device, requires_grad=False)
        self.default_dof_torques_upper_limits = torch_utils.to_torch(
            dof_props['effort'], dtype=torch.float32, device=self.device, requires_grad=False)

        # create environment and load assets
        spacing = self.cfg['env']['env_spacing']
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        num_envs_per_row = int(np.sqrt(self.num_envs))
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self.robot_handles = []
        self.env_handles = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_envs_per_row)
            # create robot instance
            robot_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "robot", i, 0, 0)
            self.gym.set_actor_dof_properties(env_handle, robot_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, robot_handle)
            self.robot_handles.append(robot_handle)
            self.env_handles.append(env_handle)

        # find link & joint names.
        base_names = ['pelvis']     # 1
        torso_names = ['torso']     # 1
        hand_names = [s for s in self.link_names if 'elbow' in s]  # 2
        shoulder_names = [s for s in self.link_names if 'shoulder' in s]    # 6
        hip_names = [s for s in self.link_names if 'hip_roll' in s or 'hip_yaw' in s]   # 4
        thigh_names = [s for s in self.link_names if 'hip_pitch' in s]  # 2
        calf_names = [s for s in self.link_names if 'knee' in s]    # 2
        foot_names = [s for s in self.link_names if 'ankle' in s]   # 2
        terminate_touch_names = shoulder_names
        undesired_touch_names = torso_names + hip_names + thigh_names + calf_names

        # find hand & foot & knee & hip & base's index
        self.torso_indices = torch.zeros(
            len(torso_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.hand_indices = torch.zeros(
            len(hand_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.shoulder_indices = torch.zeros(
            len(shoulder_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.hip_indices = torch.zeros(
            len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.thigh_indices = torch.zeros(
            len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.calf_indices = torch.zeros(
            len(calf_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.foot_indices = torch.zeros(
            len(foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.terminate_touch_indices = torch.zeros(
            len(terminate_touch_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.undesired_touch_indices = torch.zeros(
            len(undesired_touch_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(torso_names)):
            self.torso_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], torso_names[i])
        for i in range(len(hand_names)):
            self.hand_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], hand_names[i])
        for i in range(len(shoulder_names)):
            self.shoulder_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], shoulder_names[i])
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], hip_names[i])
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], thigh_names[i])
        for i in range(len(calf_names)):
            self.calf_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], calf_names[i])
        for i in range(len(foot_names)):
            self.foot_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], foot_names[i])
        for i in range(len(terminate_touch_names)):
            self.terminate_touch_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], terminate_touch_names[i])
        for i in range(len(undesired_touch_names)):
            self.undesired_touch_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.env_handles[0], self.robot_handles[0], undesired_touch_names[i])
        self.base_index = self.gym.find_actor_rigid_body_handle(
            self.env_handles[0], self.robot_handles[0], base_names[0])

    def reset(self, is_uniform_rollout=True):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # ================= for uniform rollouts ================= #
        if is_uniform_rollout:
            self.progress_buf[:] = torch.randint_like(self.progress_buf, low=0, high=self.max_episode_length)
        # ======================================================== #
        return super().reset()

    def reset_idx(self, env_ids):
        # convert env_id's dtype
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # reset dof position & velocity
        positions_offset = torch_utils.torch_rand_float(0.9, 1.1, (len(env_ids), self.num_dofs), device=self.device)
        velocities = torch_utils.torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dofs), device=self.device)
        self.dof_positions[env_ids] = self.default_dof_positions[env_ids] * positions_offset
        self.dof_velocities[env_ids] = velocities
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_states),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # reset robot's base frame pose
        self.root_states[env_ids] = self.base_init_state
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # refresh tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # reset buffers
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.fail_buf[env_ids] = 0
        self.stage_buf[env_ids] = 0.0
        self.stage_buf[env_ids, 0] = 1.0
        self.is_half_turn_buf[env_ids] = 0
        self.is_one_turn_buf[env_ids] = 0
        self.start_time_buf[env_ids] = torch_utils.torch_rand_float(0.0, 5.0, (len(env_ids), 1), device=self.device).squeeze()
        self.cmd_time_buf[env_ids] = 0.0
        self.land_time_buf[env_ids] = 0.0

        # reset inner variables
        self.prev_actions[env_ids] = 0
        self.joint_targets[env_ids] = self.dof_positions[env_ids]

        # calculate commands
        commands = torch.zeros((len(env_ids), 3), dtype=torch.float32, device=self.device)
        masks0 = (self.cmd_time_buf[env_ids] == 0).type(torch.float32)
        masks1 = (1.0 - masks0)*(self.progress_buf[env_ids]*self.control_dt < self.cmd_time_buf[env_ids] + 0.2).type(torch.float32)
        masks2 = (1.0 - masks0)*(1.0 - masks1)
        commands[:, 0] = masks0
        commands[:, 1] = masks1
        commands[:, 2] = masks2

        # reset observation
        obs = jit_compute_observations(
            self.base_quaternions[env_ids], self.world_z[env_ids], 
            self.dof_positions[env_ids], self.dof_velocities[env_ids], 
            self.prev_actions[env_ids], self.progress_buf[env_ids]*self.control_dt, self.gait_freq, commands)
        for history_idx in range(self.history_len):
            self.obs_buf[env_ids, history_idx*self.raw_obs_dim:(history_idx+1)*self.raw_obs_dim] = obs

        # reset state
        contact_forces = self.contact_forces[env_ids]
        hand_contact_forces = contact_forces[:, self.hand_indices, :]
        foot_contact_forces = contact_forces[:, self.foot_indices, :]
        self.states_buf[env_ids] = jit_compute_states(
            self.base_quaternions[env_ids], self.base_lin_vels[env_ids], self.base_ang_vels[env_ids], self.base_positions[env_ids],
            hand_contact_forces, foot_contact_forces, self.stage_buf[env_ids])

    def pre_physics_step(self, actions: torch.Tensor):
        # update previous actions
        self.prev_actions[:] = actions

        # set PD targets
        self.joint_targets[:] = self.action_smooth_weight*(actions*self.action_scale + self.default_dof_positions) \
                                                            + (1.0 - self.action_smooth_weight)*self.joint_targets
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.joint_targets))

    def post_physics_step(self):
        """
        After refresh tensors (DoF, root, contact information), need to peform followings:
        1. check termination.
        2. make observation.
        3. calculate reward.
        4. calculate costs.
        5. call 'self.reset_idx' after max_episode_len.
        6. update inner variables, such as pre_actions.
        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.progress_buf += 1

        # stage 0: stand, stage 1: down, stage 2: jump, stage 3: back turn, stage 4: land
        # =================== calculate rewards =================== #
        # com height
        jump_pos = self.base_positions + torch_utils.quat_rotate(self.base_quaternions, self.robot_up)
        jump_height = jump_pos[:, 2]
        com_height = self.base_positions[:, 2]
        self.rew_buf[:, 0]  = self.stage_buf[:, 0]*(-torch.abs(com_height - 0.95))
        self.rew_buf[:, 0] += self.stage_buf[:, 1]*(-torch.abs(com_height - 0.6))
        self.rew_buf[:, 0] += self.stage_buf[:, 2]*(jump_height <= 1.8)*(jump_height)
        self.rew_buf[:, 0] += self.stage_buf[:, 3]*(jump_height <= 1.8)*(jump_height)
        self.rew_buf[:, 0] += self.stage_buf[:, 4]*(-torch.abs(com_height - 0.95))
        # body balance
        body_z = torch_utils.quat_rotate_inverse(self.base_quaternions, self.world_z)
        self.rew_buf[:, 1] =  self.stage_buf[:, 0]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 1] += self.stage_buf[:, 1]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        self.rew_buf[:, 1] += self.stage_buf[:, 2]*(-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - np.pi/2.0))
        self.rew_buf[:, 1] += self.stage_buf[:, 3]*(-torch.abs(torch.arccos(torch.clamp(body_z[:, 1], -1.0, 1.0)) - np.pi/2.0))
        self.rew_buf[:, 1] += self.stage_buf[:, 4]*(-torch.arccos(torch.clamp(body_z[:, 2], -1.0, 1.0)))
        # pitch vel
        base_lin_vels = torch_utils.quat_rotate_inverse(self.base_quaternions, self.base_lin_vels)
        base_ang_vels = torch_utils.quat_rotate_inverse(self.base_quaternions, self.base_ang_vels)
        base_vel_penalties = torch.square(base_lin_vels[:, 0]) + torch.square(base_lin_vels[:, 1]) + torch.square(base_ang_vels[:, 2])
        self.rew_buf[:, 2]  = self.stage_buf[:, 0]*(-base_vel_penalties)
        self.rew_buf[:, 2] += self.stage_buf[:, 1]*(-base_vel_penalties)
        self.rew_buf[:, 2] += self.stage_buf[:, 2]*(1.0 - self.is_one_turn_buf)*(-base_ang_vels[:, 1])
        self.rew_buf[:, 2] += self.stage_buf[:, 3]*(1.0 - self.is_one_turn_buf)*(-base_ang_vels[:, 1])
        self.rew_buf[:, 2] += self.stage_buf[:, 4]*(-base_vel_penalties)
        # energy
        self.rew_buf[:, 3] = -torch.square(self.dof_torques).mean(dim=-1)
        # style
        self.rew_buf[:, 4]  = self.stage_buf[:, 0]*(-torch.square(self.dof_positions - self.default_dof_positions).mean(dim=-1))
        self.rew_buf[:, 4] += self.stage_buf[:, 1]*(-torch.square(self.dof_positions - self.sit_dof_positions).mean(dim=-1))
        self.rew_buf[:, 4] += self.stage_buf[:, 2]*(-torch.square(self.dof_positions[:, 10:] - self.default_dof_positions[:, 10:]).mean(dim=-1))
        self.rew_buf[:, 4] += self.stage_buf[:, 3]*(-torch.square(self.dof_positions[:, 10:] - self.default_dof_positions[:, 10:]).mean(dim=-1))
        self.rew_buf[:, 4] += self.stage_buf[:, 4]*(-torch.square(self.dof_positions - self.default_dof_positions).mean(dim=-1))
        # ========================================================= #
        # ==================== calculate costs ==================== #
        # foot contact
        foot_contact_threshold = 0.25
        foot_contact_forces = self.contact_forces[:, self.foot_indices, :]
        hand_contact_forces = self.contact_forces[:, self.hand_indices, :]
        foot_contact = (torch.norm(foot_contact_forces, dim=2) > 1.0).type(torch.float)
        left_phase = (self.obs_buf[:, -6] <= 0).to(torch.float)
        right_phase = (self.obs_buf[:, -4] <= 0).to(torch.float)
        foot_contact_cost  = left_phase*(1.0 - foot_contact[:, 0]) + (1.0 - left_phase)*foot_contact[:, 0]
        foot_contact_cost += right_phase*(1.0 - foot_contact[:, 1]) + (1.0 - right_phase)*foot_contact[:, 1]
        foot_contact_cost /= 2.0
        self.cost_buf[:, 0]  = self.stage_buf[:, 0]*foot_contact_cost
        self.cost_buf[:, 0] += self.stage_buf[:, 1]*foot_contact_cost
        self.cost_buf[:, 0] += self.stage_buf[:, 2]*foot_contact_cost
        self.cost_buf[:, 0] += self.stage_buf[:, 3]*foot_contact_threshold
        self.cost_buf[:, 0] += self.stage_buf[:, 4]*foot_contact_cost
        # body contact
        term_contact = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1)
        undesired_contact = torch.any(torch.norm(self.contact_forces[:, self.undesired_touch_indices, :], dim=-1) > 1.0, dim=-1)
        self.cost_buf[:, 1] =  self.stage_buf[:, 0]*torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 1]*torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 2]*torch.logical_or(term_contact, undesired_contact).type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 3]*undesired_contact.type(torch.float)
        self.cost_buf[:, 1] += self.stage_buf[:, 4]*undesired_contact.type(torch.float)
        # joint pos
        self.cost_buf[:, 2] = torch.mean((
            (self.dof_positions < self.dof_pos_lower_limits)|(self.dof_positions > self.dof_pos_upper_limits)
            ).to(torch.float), dim=-1)
        # joint vel
        self.cost_buf[:, 3] = torch.mean(
            (torch.abs(self.dof_velocities) > self.dof_vel_upper_limits).to(torch.float), dim=-1)
        # joint torque
        self.cost_buf[:, 4] = torch.mean(
            (torch.abs(self.dof_torques) > self.dof_torques_upper_limits).to(torch.float), dim=-1)
        # ========================================================= #

        # update stage
        # have to handle in the following order: N -> N-1 -> N-2 ... -> 1 -> 0.
        from4_to0 = torch.logical_and(self.stage_buf[:, 4] == 1.0, 
            self.progress_buf*self.control_dt > self.land_time_buf + 0.2).type(torch.float32)
        self.stage_buf[:, 4] = (1.0 - from4_to0)*self.stage_buf[:, 4]
        self.stage_buf[:, 0] = from4_to0 + (1.0 - from4_to0)*self.stage_buf[:, 0]
        from3_to4 = torch.logical_and(
            self.stage_buf[:, 3] == 1.0, torch.logical_and(
                foot_contact.mean(dim=-1) > 0.0,
                self.is_half_turn_buf
            )
        ).type(torch.float32)
        self.stage_buf[:, 3] = (1.0 - from3_to4)*self.stage_buf[:, 3]
        self.stage_buf[:, 4] = from3_to4 + (1.0 - from3_to4)*self.stage_buf[:, 4]
        from2_to3 = torch.logical_and(
            self.stage_buf[:, 2] == 1.0, torch.logical_and(
                foot_contact.mean(dim=-1) < 0.1,
                com_height >= 0.8)).type(torch.float32)
        self.stage_buf[:, 2] = (1.0 - from2_to3)*self.stage_buf[:, 2]
        self.stage_buf[:, 3] = from2_to3 + (1.0 - from2_to3)*self.stage_buf[:, 3]
        from1_to2 = torch.logical_and(
            self.stage_buf[:, 1] == 1.0, torch.logical_and(
                com_height <= 0.7, foot_contact.mean(dim=-1) == 1.0)).type(torch.float32)
        self.stage_buf[:, 1] = (1.0 - from1_to2)*self.stage_buf[:, 1]
        self.stage_buf[:, 2] = from1_to2 + (1.0 - from1_to2)*self.stage_buf[:, 2]
        from0_to1 = torch.logical_and(
            self.stage_buf[:, 0] == 1.0, torch.logical_and(
                self.progress_buf*self.control_dt > self.start_time_buf, torch.logical_and(
                    com_height >= 0.8, self.is_half_turn_buf == 0))).type(torch.float32)
        self.stage_buf[:, 0] = (1.0 - from0_to1)*self.stage_buf[:, 0]
        self.stage_buf[:, 1] = from0_to1 + (1.0 - from0_to1)*self.stage_buf[:, 1]

        # check the robot tumbling
        self.is_half_turn_buf[:] = torch.logical_or(
            self.is_half_turn_buf, torch.logical_and(
                base_ang_vels[:, 1] < 0, torch.logical_and(
                    body_z[:, 2] < 0, body_z[:, 0] < 0))).type(torch.long)
        self.is_one_turn_buf[:] = torch.logical_or(
            self.is_one_turn_buf, torch.logical_and(
                self.is_half_turn_buf, torch.logical_and(
                    body_z[:, 0] >= 0, body_z[:, 2] >= 0))).type(torch.long)
        land_masks = torch.logical_and(self.land_time_buf == 0, self.stage_buf[:, 4] == 1).type(torch.float32)
        self.land_time_buf[:] = land_masks*(self.progress_buf*self.control_dt) + (1.0 - land_masks)*self.land_time_buf
        cmd_masks = torch.logical_and(self.cmd_time_buf == 0, self.stage_buf[:, 1] == 1).type(torch.float32)
        self.cmd_time_buf[:] = cmd_masks*(self.progress_buf*self.control_dt) + (1.0 - cmd_masks)*self.cmd_time_buf

        # check termination
        body_contacts = torch.any(torch.norm(self.contact_forces[:, self.terminate_touch_indices, :], dim=-1) > 1.0, dim=-1)
        landing_wo_turns = torch.logical_and(self.stage_buf[:, 3] == 1.0, torch.logical_and(foot_contact.mean(dim=-1) > 0.0, 1 - self.is_half_turn_buf))
        body_balances = torch.logical_and(self.stage_buf[:, 0] + self.stage_buf[:, 1] + self.stage_buf[:, 2] >= 1.0, body_z[:, 2] < 0.5)
        self.fail_buf[:] = torch.logical_or(body_contacts, torch.logical_or(landing_wo_turns, body_balances)).type(torch.long)

        # calculate reset buffer        
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self.max_episode_length, 
            torch.ones_like(self.reset_buf), self.fail_buf
        )

        # calculate commands
        commands = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        masks0 = (self.cmd_time_buf == 0).type(torch.float32)
        masks1 = (1.0 - masks0)*(self.progress_buf*self.control_dt < self.cmd_time_buf + 0.2).type(torch.float32)
        masks2 = (1.0 - masks0)*(1.0 - masks1)
        commands[:, 0] = masks0
        commands[:, 1] = masks1
        commands[:, 2] = masks2

        # update observation buffer
        obs = jit_compute_observations(
            self.base_quaternions, self.world_z, 
            self.dof_positions, self.dof_velocities, 
            self.prev_actions, self.progress_buf*self.control_dt, self.gait_freq, commands)
        self.obs_buf[:, :-self.raw_obs_dim] = self.obs_buf[:, self.raw_obs_dim:].clone()
        self.obs_buf[:, -self.raw_obs_dim:] = obs

        # update state buffer
        self.states_buf[:] = jit_compute_states(
            self.base_quaternions, self.base_lin_vels, self.base_ang_vels, self.base_positions,
            hand_contact_forces, foot_contact_forces, self.stage_buf)

        # return extra
        self.extras['costs'] = self.cost_buf.clone()
        self.extras['fails'] = self.fail_buf.clone()
        self.extras['next_obs'] = self.obs_buf.clone()
        self.extras['next_states'] = self.states_buf.clone()
        self.extras['dones'] = self.reset_buf.clone()

        # reset
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0: self.reset_idx(env_ids)

@torch.jit.script
def jit_compute_states(
    base_quaternions, base_lin_vels, base_ang_vels, base_positions, 
    hand_contact_forces, foot_contact_forces, stages,
):
    bb_lin_vels = torch_utils.quat_rotate_inverse(base_quaternions, base_lin_vels)
    bb_ang_vels = torch_utils.quat_rotate_inverse(base_quaternions, base_ang_vels)
    com_height = base_positions[:, 2:3]
    hand_contacts = (torch.norm(hand_contact_forces, dim=2) > 1.0).type(torch.float)
    foot_contacts = (torch.norm(foot_contact_forces, dim=2) > 1.0).type(torch.float)
    states = torch.cat([
        bb_lin_vels, bb_ang_vels, com_height, hand_contacts, foot_contacts, stages], dim=-1)
    return states

@torch.jit.script
def jit_compute_observations(
    base_quat, world_z, dof_pos, dof_vel, 
    prev_actions, world_time, gait_freq, command,
):
    obs_list = []
    # body orientation
    obs_list.append(torch_utils.quat_rotate_inverse(base_quat, world_z)) # 3
    # DoF's position and velocity
    obs_list.append(dof_pos) # 19
    obs_list.append(dof_vel) # 19
    # previous actions
    obs_list.append(prev_actions) # 19
    # central phase
    obs_list.append(command[:, 0:1]*torch.cos(2.0*np.pi*gait_freq*world_time.unsqueeze(dim=-1)))
    obs_list.append(command[:, 0:1]*torch.sin(2.0*np.pi*gait_freq*world_time.unsqueeze(dim=-1)))
    obs_list.append(command[:, 0:1]*torch.cos(2.0*np.pi*gait_freq*world_time.unsqueeze(dim=-1) + np.pi))
    obs_list.append(command[:, 0:1]*torch.sin(2.0*np.pi*gait_freq*world_time.unsqueeze(dim=-1) + np.pi))
    # command
    obs_list.append(command) # 3
    # concatenate
    obs = torch.cat(obs_list, dim=-1) # 3 + 19*3 + 4 + 3
    return obs
