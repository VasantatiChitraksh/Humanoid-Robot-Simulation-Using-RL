import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import os


class HumanoidWalkEnv(gym.Env):
    """
    Custom PyBullet Environment for Humanoid Locomotion,
    compliant with the Gymnasium API.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):

        self.render_mode = render_mode
        self.physics_client = None

        # --- 1. Start PyBullet ---
        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF("plane.urdf")
        # --- 2. Load Assets & Define Robot ---
        p.setGravity(0, 0, -9.8)
        self.urdf_path = "humanoid/humanoid.urdf"
        self.robot_id = p.loadURDF(
            self.urdf_path, [0, 0, 1.5], useFixedBase=False)

        # --- 3. Build Joint List (CRITICAL) ---
        self.all_joint_indices = list(range(p.getNumJoints(self.robot_id)))
        self.actuated_joints_info = []

        revolute_joint_count = 0
        spherical_joint_count = 0

        for j in self.all_joint_indices:
            info = p.getJointInfo(self.robot_id, j)
            joint_type = info[2]

            if joint_type != p.JOINT_FIXED:
                self.actuated_joints_info.append({
                    'name': info[1].decode('UTF-8'),
                    'index': info[0],
                    'type': joint_type
                })
                if joint_type == p.JOINT_REVOLUTE:
                    revolute_joint_count += 1
                elif joint_type == p.JOINT_SPHERICAL:
                    spherical_joint_count += 1

        self.actuated_joint_names = [info['name']
                                     for info in self.actuated_joints_info]
        self.num_actuated_joints = len(self.actuated_joint_names)

        print("--- Actuated Joints Found ---")
        print(self.actuated_joint_names)

        # --- 4. Define Spaces ---
        # State: torso pos (3), torso orn (4), joint states
        # Joint states = (revolute_pos * 1) + (revolute_vel * 1)
        #              + (spherical_pos * 4) + (spherical_vel * 3)
        state_dim = 7 + (revolute_joint_count * 2) + \
            (spherical_joint_count * 7)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        self.action_bins = 5
        self.action_space = gym.spaces.MultiDiscrete(
            [self.action_bins] * self.num_actuated_joints)

    def _get_observation(self):
        """Helper to get the current state of the robot."""

        # Get Torso State
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot_id)

        # Get Joint States
        joint_states = p.getJointStates(
            self.robot_id, [info['index'] for info in self.actuated_joints_info])

        joint_positions = []
        joint_velocities = []

        for i, info in enumerate(self.actuated_joints_info):
            if info['type'] == p.JOINT_REVOLUTE:
                joint_positions.append(joint_states[i][0])
                joint_velocities.append(joint_states[i][1])
            elif info['type'] == p.JOINT_SPHERICAL:
                joint_positions.append(joint_states[i][0])
                joint_velocities.append(joint_states[i][1])

        return np.concatenate([
            torso_pos,
            torso_orn,
            joint_positions,
            joint_velocities
        ]).astype(np.float32)

    def _calculate_reward(self, torso_pos):
        current_x = torso_pos[0]
        r_vel = current_x - self.last_x
        self.last_x = current_x

        is_alive = torso_pos[2] > 0.8
        r_live = 0.01 if is_alive else 0.0
        r_energy = 0.0  # TODO: Add energy penalty in M3

        if not is_alive:
            r_live = -1.0

        return r_vel + r_live - r_energy

    def step(self, action):

        # TODO: Implement action mapping in Module 3
        # (e.g., convert [0, 2, 4] to torques [-1, 0, 1])

        p.stepSimulation()

        torso_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        observation = self._get_observation()
        reward = self._calculate_reward(torso_pos)

        terminated = torso_pos[2] < 0.8  # Fall check
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None, initial_pose: np.ndarray = None):
        """
        Resets the simulation and sets the robot to the
        provided initial_pose.
        """
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)

        # --- CAMERA (Side View) ---
        p.resetDebugVisualizerCamera(
            cameraDistance=2.5,  # Closer
            cameraYaw=90,        # Side view
            cameraPitch=-10,     # Slightly looking down
            cameraTargetPosition=[0, 0, 1.0]  # Aimed at the torso
        )

        self.robot_id = p.loadURDF(
            self.urdf_path, [0, 0, 1.5], useFixedBase=False)

        self.plane_id = p.loadURDF("plane.urdf")

        if initial_pose is not None:
            if len(initial_pose) != self.num_actuated_joints:
                raise ValueError(
                    f"Pose vector length ({len(initial_pose)}) does not "
                    f"match robot's actuated joints ({self.num_actuated_joints})."
                )

            # --- THIS IS THE CORRECT, UN-COMMENTED LOGIC ---
            for i, joint_name in enumerate(self.actuated_joint_names):
                joint_info = self.actuated_joints_info[i]
                joint_index = joint_info['index']
                joint_type = joint_info['type']

                angle = initial_pose[i]

                if joint_type == p.JOINT_REVOLUTE:
                    # Use resetJointState for 1-DOF joints
                    p.resetJointState(
                        self.robot_id,
                        joint_index,
                        targetValue=angle,
                        targetVelocity=0.0
                    )

                elif joint_type == p.JOINT_SPHERICAL:
                    # Use resetJointStateMultiDof for Multi-DOF joints
                    # We assume our 2D angle maps to a "pitch" rotation
                    quaternion = p.getQuaternionFromEuler([angle, 0, 0])

                    p.resetJointStateMultiDof(
                        self.robot_id,
                        joint_index,
                        targetValue=quaternion,   # Pass the 4-value quaternion
                        targetVelocity=[0, 0, 0]  # Pass a 3-value list
                    )
            # ---------------------------------------------------

        self.last_x = 0.0

        observation = self._get_observation()
        info = {}

        return observation, info

    def render(self):
        pass

    def close(self):
        if self.physics_client is not None:
            try:
                p.disconnect()
                self.physics_client = None
            except p.error:
                pass
