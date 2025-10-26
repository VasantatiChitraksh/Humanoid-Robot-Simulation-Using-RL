import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
import os


class HumanoidWalkEnv(gym.Env):
    """Custom PyBullet Environment for Humanoid Locomotion, compliant with the Gymnasium API."""

    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.physics_client = None

        if self.render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

        self.urdf_path = "humanoid/humanoid.urdf"
        start_pos = [0, 0, 1.5]
        start_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0])
        self.robot_id = p.loadURDF(
            self.urdf_path, start_pos, start_orientation, useFixedBase=False)

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

        print("Actuated Joints Found:")
        print(self.actuated_joint_names)
        print(f"Robot loaded at position: {start_pos}")
        print(f"Robot orientation (quaternion): {start_orientation}")

        state_dim = 7 + (revolute_joint_count * 2) + \
            (spherical_joint_count * 7)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

        self.action_bins = 5
        self.action_space = gym.spaces.MultiDiscrete(
            [self.action_bins] * self.num_actuated_joints)

    def _get_observation(self):
        """Helper to get the current state of the robot."""
        torso_pos, torso_orn = p.getBasePositionAndOrientation(self.robot_id)

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

    def _calculate_reward(self, torso_pos, applied_torques):
        """
        Calculates reward based on forward velocity, staying upright,
        and energy usage. Now includes renergy.
        """
        current_x = torso_pos[0]
        r_vel = current_x - self.last_x
        self.last_x = current_x

        is_alive = torso_pos[2] > 0.8
        r_live = 0.01 if is_alive else -1.0

        energy_weight = 0.0001
        r_energy = energy_weight * np.sum(np.square(applied_torques))
        reward = 1.0 * r_vel + 1.0 * r_live - 1.0 * r_energy

        return reward

    def step(self, action_torques):
        """
        Applies torques to joints, steps simulation, and returns results.
        Accepts the actual torque values, not bin indices.
        """
        action_torques = np.array(action_torques, dtype=np.float32)

        if len(action_torques) != self.num_actuated_joints:
            raise ValueError(
                f"Action dimension ({len(action_torques)}) != num actuated joints ({self.num_actuated_joints})")

        p.setJointMotorControlArray(
            bodyIndex=self.robot_id,
            jointIndices=[info['index'] for info in self.actuated_joints_info],
            controlMode=p.TORQUE_CONTROL,
            forces=action_torques
        )
        p.stepSimulation()

        torso_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        observation = self._get_observation()
        reward = self._calculate_reward(torso_pos, action_torques)

        terminated = torso_pos[2] < 0.8
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None, initial_pose: np.ndarray = None):
        """Resets the simulation and sets the robot to the provided initial_pose."""
        super().reset(seed=seed)

        if self.physics_client is None:
            if self.render_mode == 'human':
                self.physics_client = p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            else:
                self.physics_client = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setGravity(0, 0, -9.8)

        self.plane_id = p.loadURDF("plane.urdf")

        start_pos = [0, 0, 3.5]
        start_orientation = p.getQuaternionFromEuler([np.pi/2, 0, 0])
        self.robot_id = p.loadURDF(
            self.urdf_path, start_pos, start_orientation, useFixedBase=False)

        if initial_pose is not None:
            if len(initial_pose) != self.num_actuated_joints:
                raise ValueError(
                    f"Pose vector length ({len(initial_pose)}) does not "
                    f"match robot's actuated joints ({self.num_actuated_joints})."
                )

            for i, joint_name in enumerate(self.actuated_joint_names):
                joint_info = self.actuated_joints_info[i]
                joint_index = joint_info['index']
                joint_type = joint_info['type']
                angle = initial_pose[i]

                if joint_type == p.JOINT_REVOLUTE:
                    p.resetJointState(
                        self.robot_id, joint_index,
                        targetValue=angle, targetVelocity=0.0
                    )

                elif joint_type == p.JOINT_SPHERICAL:
                    if joint_name in ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']:
                        quaternion = p.getQuaternionFromEuler([0, angle, 0])
                    elif joint_name in ['left_ankle', 'right_ankle']:
                        quaternion = p.getQuaternionFromEuler([0, 0, angle])
                        print(f"DEBUG: Using YAW mapping for {joint_name}")
                    else:
                        quaternion = p.getQuaternionFromEuler([angle, 0, 0])
                        print(f"DEBUG: Using PITCH mapping for {joint_name}")

        settling_steps = 300
        for _ in range(settling_steps):
            p.stepSimulation()

        self.last_x = p.getBasePositionAndOrientation(self.robot_id)[0][0]

        observation = self._get_observation()
        info = {}

        return observation, info

    def render(self):
        """Renders the environment (handled by PyBullet GUI)."""
        pass

    def close(self):
        """Closes the PyBullet connection."""
        if self.physics_client is not None:
            try:
                p.disconnect()
                self.physics_client = None
            except p.error:
                pass
