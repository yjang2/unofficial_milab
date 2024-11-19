import torch
import os
import re
import trimesh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from recon_utils import standard_dh_transform
import gymnasium
from gymnasium.spaces import Discrete, Box
from datetime import datetime
import imageio

class ObjectLoader:
    def __init__(self, path, dirname):
        self.path = path
        self.dirname = dirname
        self.obj_path = None
        self.voxel_grid = None
        self.image_folder = "/home/milab/Desktop/gflownet/images/"

    def _calling_target_positions(self):
        data = pd.read_csv(self.path, delimiter='\t')

        # Extract the X, Y, Z coordinates
        voxel_positions = data[['X (m)', 'Y (m)', 'Z (m)']]

        np_voxel = np.array(voxel_positions)

        return np_voxel

    def get_voxel_info(self):
        """Get information about the voxel grid."""
        if self.voxel_grid is None:
            raise ValueError("No voxel grid has been created. Please call obj_to_voxels() first.")
        
        # Number of voxels
        num_voxels = self.voxel_grid.points.shape[0]
        
        # Center of the voxel points
        center = self.voxel_grid.points.mean(axis=0)
        
        # Voxel size
        voxel_size = self.voxel_grid.pitch
        
        # Voxel positions
        voxel_pos = self.voxel_grid.points
        
        # Normalized positions with voxel size 1x1x1
        normalized_positions = self.voxel_grid.points / voxel_size
        
        return normalized_positions
    
    def render(self, joint_positions ):
        # Extract the x, y, z positions for plotting
        x_positions = joint_positions[:, 0]
        y_positions = joint_positions[:, 1]
        z_positions = joint_positions[:, 2]

        # Plot the joint positions and the links
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x_positions, y_positions, z_positions, '-o', label='Joints & Links', color='blue')
        ax.scatter(x_positions[-1], y_positions[-1], z_positions[-1], color='red', s=100, label='End-Effector')

        # print(np.min(norm_pos, axis=0))
        voxel_x = np.min(x_positions)
        # print(voxel_x)
        voxel_y = np.min(y_positions)
        voxel_z = np.min(z_positions)
        ax.scatter(voxel_x, voxel_y, voxel_z, color='g', alpha=1.0, label='Voxel Grid')
        
        # Set plot labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Robotic Arm (Forward Kinematics)')
        ax.legend()

        plt.show()

class ReplayBuffer:
    def __init__(self):
        self.visited_positions = []

    def add(self, position):
        self.visited_positions.append(position.tolist())

    def has_visited(self, position):
        return any(np.allclose(position, pos) for pos in self.visited_positions)

    def pop(self):
        """Remove the last visited position from the buffer."""
        if self.visited_positions:
            self.visited_positions.pop()

class GFNrecon(gymnasium.Env):
    def __init__(self, config):
        super(GFNrecon, self).__init__()
        self.t = 0
        self.reward = 0
        self.image_folder = "/home/milab/Desktop/gflownet/images/"
        self.frames = [] 
        self.attempts_per_joint = 0
        self.max_attempts_per_joint = 3
        self.previous_actions = []
        self.collision_history = []
        self.collision_occurred = False

        self.loader = ObjectLoader(config['target_coor_path'], config['dirname'])
        self.voxel_grid = self.loader._calling_target_positions()
        
        self.start_position = np.min(self.voxel_grid, axis=0)
        self.farthest_point = self.get_farthest_point()
    
        self.goal_tolerance = 0.1
        self.goal_reward = 20.0
        self.collision_penalty = 1.0
        
        self.current_position = self.start_position.copy()
        self.dh_params = []
        self.replay_buffer = ReplayBuffer()
        self.alpha_comb = [0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0,
                           0,0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0,
                           0,0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0,
                           0,0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0]
        self.alpha_comb = [-90 if x == 1 else x for x in self.alpha_comb]
        self.visited_positions = []
        self.collision_positions = []
        # self.visited_positions = [self.current_position.tolist()]
        # self.replay_buffer.add(self.current_position)

        self.n_joints = len(self.alpha_comb)
        num_dh_params = self.n_joints*4
        self.action_space = Discrete(3)

        # observation_space_low = np.concatenate((
        #     np.full(self.n_joints * 3, -np.inf),  # Joint positions (n_joints * 3)
        #     np.full(3, -np.inf),  # End-effector position (3)
        #     np.full(len(self.voxel_grid.flatten()), -np.inf),  # Target voxel grid points
        #     np.full(3, -np.inf),  # Action mask (3)
        #     [-np.inf],  # Number of joints left (1)
        #     np.full(3, -np.inf) # Last collision position (3)
        # ))

        # observation_space_high = np.concatenate((
        #     np.full(self.n_joints  * 3, np.inf),  # Joint positions (n_joints * 3)
        #     np.full(3, np.inf),  # End-effector position (3)
        #     np.full(len(self.voxel_grid.flatten()), np.inf),  # Target voxel grid points
        #     np.full(3, np.inf),  # Action mask (3)
        #     [np.inf],  # Number of joints left (1)
        #     np.full(3, np.inf) # Last collision position (3)
        # ))

        # self.observation_space = Box(
        #     low=observation_space_low,
        #     high=observation_space_high,
        #     dtype=np.float32
        # )

    def reset(self):
        self.t = 0
        self.reward = 0
        self.dh_params = []
        self.current_position = self.start_position.copy()
        self.visited_positions = []
        self.replay_buffer = ReplayBuffer()  # Clear the buffer
        self.collision_positions = []
        self.collision_history = []
        self.collision_occurred = False
        return self.get_state()

    def step(self, action, render = False):
        done = False
        early_termination = False      
        angle = self.apply_action(action)
        self.previous_actions.append(action)

        self.new_dh_param = {'a': 1, 'alpha': self.alpha_comb[len(self.visited_positions)], 'd': 0, 'theta': angle}

        joint_positions = self.transformation(self.dh_params + [self.new_dh_param])
        end_effector_position = np.around(joint_positions[-1], 1)
        
        tolerance = 1e-5
        end_effector_position = np.where(np.abs(end_effector_position) < tolerance, 0, end_effector_position)
        
        collision_detected = self.replay_buffer.has_visited(end_effector_position)

        if collision_detected:
            self.collision_history.append(1)
            self.collision_occurred = True
        else:
            self.collision_history.append(0)

        reward, done = self.compute_reward(end_effector_position, self.collision_occurred)

        self.dh_params.append(self.new_dh_param)  # Append DH parameters
        self.current_position = end_effector_position
        self.visited_positions.append(self.current_position.tolist())
        self.replay_buffer.add(end_effector_position)
        self.render(joint_positions, render)
        self.t += 1

        if len(self.visited_positions) >= len(self.alpha_comb):
            done = True

        return self.get_state(early_termination), reward, done, {}
    
    def get_state(self, early_termination=False):
        joint_features = self.get_joint_features()  # [n_joints * 3]
        
        target_points = self.get_target_joints()  # [n_voxels, 3]

        collision_history = self.get_collision_history()
        
        if len(self.dh_params) == 0:
            end_effector_positions = self.start_position  # Shape [1, 3], with initial zeros
        else:
            positions = self.transformation(self.dh_params)  # Shape: [t, 3]
            end_effector_positions = positions[-1]
            # print(f"end_effector_position: {end_effector_positions}")
        
        if end_effector_positions.ndim == 1:
            end_effector_positions = end_effector_positions.reshape(1, 3)
        state_dict = {
            "joint_features": joint_features,
            "target_points": target_points,    # For encoder: [n_voxels, 3]
            "collision_history": collision_history,
            "end_effector_positions": end_effector_positions
        }
        
        return state_dict
    
    # def compute_reward(self, current_position, collisions):
    #     target_positions = self.voxel_grid
    #     done = False
    #     reward = 0.0

    #     # Calculate the boundaries based on target_positions (forming a cube)
    #     x_min, y_min, z_min = target_positions.min(axis=0)
    #     x_max, y_max, z_max = target_positions.max(axis=0)
    
    #     within_boundary = (
    #     x_min <= current_position[0] < x_max and
    #     y_min <= current_position[1] < y_max and
    #     z_min <= current_position[2] < z_max
    #     )
        
    #     if np.any(np.all(np.isclose(target_positions, current_position, atol=0.1), axis=1)):
    #         reward += 0.5
    #     elif not within_boundary:
    #         reward -= 0.1
    #     # if np.all(np.isclose(target_positions, current_position, atol=1e-1)):
    #     #     reward += 20.  # larger reward for completing the task
    
    #     # reward = max(-0.01, reward)
    #     return reward, done
    
    def compute_reward(self, current_position, collisions):
        target_positions = self.voxel_grid
        done = False
        reward = 0
        # Compute distances
        distance_to_farthest = np.linalg.norm(current_position - self.farthest_point)
        max_distance = np.linalg.norm(self.start_position - self.farthest_point)
        
        # Penalize collisions (ensure penalty doesn't make reward negative)
        if collisions:
            reward -= self.collision_penalty
            reward = max(reward, 1e-8)
        else:  
            # Calculate the reward as the difference from max distance
            reward += max_distance - distance_to_farthest  # Reward increases as distance decreases
            # Ensure reward is positive
            reward = max(reward, 1e-8)
            
            if np.any(np.all(np.isclose(target_positions, current_position, atol=0.1), axis=1)):
                reward += 5.0
            
            # Check if the end-effector has reached the farthest point within tolerance
            if distance_to_farthest < self.goal_tolerance:
                reward += self.goal_reward  # Provide a significant reward for reaching the goal
        return reward, done



    def get_joint_features(self):
        joint_positions = np.zeros((self.n_joints, 3), dtype=np.float32)
        if len(self.dh_params) == 0:
            joint_positions[0] = self.start_position
        else:
            positions = self.transformation(self.dh_params)
            positions = positions[1:]  # Exclude the base position
            joint_positions[:len(positions)] = positions
        return joint_positions
        
    def get_target_joints(self):
        target_points = self.voxel_grid
        return target_points.astype(np.float32)

    def get_collision_history(self):
        collision_history = np.zeros(self.n_joints, dtype=np.float32)
        collision_history[:len(self.collision_history)] = np.array(self.collision_history, dtype=np.float32)
        return collision_history
    
    def end_effector_pos(self):
        if len(self.dh_params) == 0:
            end_effector_positions = self.start_position  # Shape [1, 3], with initial zeros
        else:
            positions = self.transformation(self.dh_params)  # Shape: [t, 3]
            end_effector_positions = positions[-1]
            # print(f"end_effector_position: {end_effector_positions}")
        
        if end_effector_positions.ndim == 1:
            end_effector_positions = end_effector_positions.reshape(1, 3)
        return end_effector_positions

    def transformation(self, dh_params):
        T_0_7 = np.eye(4)
        T_0_7[:3, 3] = self.start_position

        joint_positions = [T_0_7[:3, 3].tolist()]

        for params in dh_params:
            A_i = standard_dh_transform(params['a'], params['alpha'], params['d'], params['theta'])
            T_0_7 = np.dot(T_0_7, A_i)
            joint_positions.append(T_0_7[:3, 3].tolist())

        joint_positions = np.array(joint_positions)
        return joint_positions[1:]

    def apply_action(self, action):
        """Apply an action that rotates the joint to a specific angle."""
        angles = {
            0: -90.0,  # Rotate -90 degrees
            1: 0.0,    # No rotation
            2: 90.0,   # Rotate +90 degrees
        }
        return angles.get(action, 0.0)
    
    def render(self, joint_positions, total_reward, save_gif = False):
        if save_gif:
            joint_positions = np.insert(joint_positions, 0, self.start_position, axis=0)
            """Render the current path of the agent."""
            x_positions = joint_positions[:, 0]
            y_positions = joint_positions[:, 1]
            z_positions = joint_positions[:, 2]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            voxel_x = self.voxel_grid[:, 0]
            voxel_y = self.voxel_grid[:, 1]
            voxel_z = self.voxel_grid[:, 2]
            ax.scatter(voxel_x, voxel_y, voxel_z, color='gray', alpha=0.5, label='Voxel Grid')

            ax.plot(x_positions, y_positions, z_positions, '-o', label='Joints & Links', color='blue')
            ax.scatter(x_positions[-1], y_positions[-1], z_positions[-1], color='red', s=100, label='End-Effector')
        
            self.visualize_candidates(ax, joint_positions[-1])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Path Finding #{self.t} with Total Reward: {total_reward:.4f}')
            ax.legend()

            date = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{self.image_folder}robot{self.t}.png"
            plt.savefig(filename)
            self.frames.append(filename)  # Store the filename for GIF creation
            if self.t >= len(self.alpha_comb):  # Create GIF at the end of the episode
                gif_filename = f"{self.image_folder}reuslt_{date}.gif"
                self.create_gif(gif_filename)
                print(f"GIF saved as {gif_filename}")

    def all_moves_taken(self, current_position):
        """Check if all potential moves for the end-effector are already visited."""
        candidates = []
        for action in range(3):
            angle = self.apply_action(action)
            alpha_index = min(len(self.dh_params), len(self.alpha_comb) - 1)
            new_dh_param = {'a': 1, 'alpha': self.alpha_comb[alpha_index], 'd': 0, 'theta': angle}
            joint_positions = self.transformation(self.dh_params + [new_dh_param])
            candidate_position = np.around(joint_positions[-1], 1)

            if not self.replay_buffer.has_visited(candidate_position):
                candidates.append(candidate_position)

        return len(candidates) == 0

    def visualize_candidates(self, ax, end_effector_position):
        candidates = []
        for action in range(3):
            angle = self.apply_action(action)
            alpha_index = min(len(self.visited_positions), len(self.alpha_comb) - 1)
            new_dh_param = {'a': 1, 'alpha': self.alpha_comb[alpha_index], 'd': 0, 'theta': angle}
            joint_positions = self.transformation(self.dh_params + [new_dh_param])
            candidate_position = np.around(joint_positions[-1], 1)
            if not self.replay_buffer.has_visited(candidate_position):
                candidates.append(candidate_position)

        candidates = np.array(candidates)

        # Visualize the candidates
        ax.scatter(candidates[:, 0], candidates[:, 1], candidates[:, 2], color='green', s=50, alpha = 1.0,label='Candidate Positions')
        ax.legend()

        # Check if the current end-effector position is on a target voxel
        # if np.any(np.all(np.isclose(self.voxel_grid, end_effector_position, atol=0.1), axis=1)):
        #     print(f"End-effector reached a target voxel at position: {end_effector_position}")

        return candidates

    def create_gif(self, gif_filename):
        """Create a GIF from saved frames."""
        with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
            for frame in self.frames:
                image = imageio.imread(frame)
                writer.append_data(image)
        self.frames.clear()
    
    def get_farthest_point(self):
        # Compute distances from start_position to all target points
        distances = np.linalg.norm(self.voxel_grid - self.start_position, axis=1)
        # Get the index of the farthest point
        idx = np.argmax(distances)
        # Return the farthest point
        return self.voxel_grid[idx]