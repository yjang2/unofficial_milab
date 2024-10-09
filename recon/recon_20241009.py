import torch
import os
import re
import trimesh
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
        self.image_folder = "/home/milab/Downloads/gflownet-trunk/src/gflownet/images/"

    def _calling_obj(self, index=0):
        """Load .obj files from the specified directory."""
        all_files = []
        target_dir = os.path.join(self.path, self.dirname)
        if not os.path.isdir(target_dir):
            raise ValueError(f"The directory {target_dir} does not exist.")
        for file in sorted(os.listdir(target_dir), key=lambda x: int(re.search(r'(\d+)', x).group() if re.search(r'(\d+)', x) else 0)):
            if file.endswith('.obj'):
                all_files.append(os.path.join(target_dir, file))
        
        if not all_files:
            raise ValueError(f"No .obj files found in the directory {target_dir}.")
        
        if index < 0 or index >= len(all_files):
            raise ValueError(f"Index out of range. There are {len(all_files)} .obj files but index {index} was provided.")
        
        self.obj_path = all_files[index]
        print(f"Loading file: {self.obj_path}")
        return self.obj_path

    def obj_to_voxels(self, target_voxel_count):
        """Convert the loaded OBJ file to a voxel grid."""
        if self.obj_path is None:
            raise ValueError("No OBJ file has been loaded. Please call _calling_obj() first.")
        
        # Load the OBJ file using trimesh
        mesh = trimesh.load(self.obj_path)
        
        # Estimate an initial voxel resolution
        bounding_box = mesh.bounds
        size = bounding_box[1] - bounding_box[0]
        
        # Assume a cubic voxel grid and calculate resolution to match the target voxel count
        volume = np.prod(size)
        voxel_size = (volume / target_voxel_count) ** (1/3)
        approximate_resolution = size / voxel_size
        
        # Create a voxelized version of the mesh
        voxels = mesh.voxelized(pitch=voxel_size)
        
        # Refine the resolution to better match the target number of voxels
        while voxels.points.shape[0] > target_voxel_count:
            voxel_size *= 1.05
            voxels = mesh.voxelized(pitch=voxel_size)
            
        while voxels.points.shape[0] < target_voxel_count:
            voxel_size /= 1.05
            voxels = mesh.voxelized(pitch=voxel_size)
        
        self.voxel_grid = voxels
        return self.voxel_grid

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
        self.image_folder = "/home/milab/Downloads/gflownet-trunk/src/gflownet/images/"
        self.frames = [] 
        self.attempts_per_joint = 0
        self.max_attempts_per_joint = 3
        self.previous_actions = []
        self.collision_history = []

        # Initialize ObjectLoader with config
        self.loader = ObjectLoader(config['target_path'], config['dirname'])
        self.loader._calling_obj(config.get('obj_index', 0))  # Load specified .obj file
        self.loader.obj_to_voxels(config.get('target_voxel_count', 63))
        self.voxel_grid = self.loader.get_voxel_info()  # Get voxel positions
        
        # Initialize start position and other variables
        self.start_position = np.min(self.voxel_grid, axis=0)
        
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

        observation_space_low = np.concatenate((
            np.full(self.n_joints * 3, -np.inf),  # Joint positions (n_joints * 3)
            np.full(3, -np.inf),  # End-effector position (3)
            np.full(len(self.voxel_grid.flatten()), -np.inf),  # Target voxel grid points
            np.full(3, -np.inf),  # Action mask (3)
            [-np.inf],  # Number of joints left (1)
            np.full(3, -np.inf) # Last collision position (3)
        ))

        observation_space_high = np.concatenate((
            np.full(self.n_joints  * 3, np.inf),  # Joint positions (n_joints * 3)
            np.full(3, np.inf),  # End-effector position (3)
            np.full(len(self.voxel_grid.flatten()), np.inf),  # Target voxel grid points
            np.full(3, np.inf),  # Action mask (3)
            [np.inf],  # Number of joints left (1)
            np.full(3, np.inf) # Last collision position (3)
        ))

        self.observation_space = Box(
            low=observation_space_low,
            high=observation_space_high,
            dtype=np.float32
        )
        # print(f"Observation space target points shape: {self.voxel_grid.shape}")  # Should be (n_voxels, 3)
    def reset(self):
        self.t = 0
        self.reward = 0
        self.dh_params = []
        self.current_position = self.start_position.copy()
        self.visited_positions = []
        # self.visited_positions = [self.current_position.tolist()]
        self.replay_buffer = ReplayBuffer()  # Clear the buffer
        self.collision_positions = []
        # self.replay_buffer.add(self.current_position)
        return self.get_state()

    def step(self, action, render = False):
        early_termination = False      
        angle = self.apply_action(action)
        self.previous_actions.append(action)
        # print(f"t: {self.t}, angle: {angle}")

        # Create the new DH parameter based on the action
        self.new_dh_param = {'a': 1, 'alpha': self.alpha_comb[len(self.visited_positions)], 'd': 0, 'theta': angle}

        # Apply transformations to get new joint positions
        joint_positions = self.transformation(self.dh_params + [self.new_dh_param])
        
        end_effector_position = np.around(joint_positions[-1], 1)
        
        # Apply a small tolerance to treat values very close to zero as zero
        tolerance = 1e-5
        end_effector_position = np.where(np.abs(end_effector_position) < tolerance, 0, end_effector_position)
        
        # Check for collisions or if the position has been visited
        collision_detected = self.replay_buffer.has_visited(end_effector_position)
        reward, done = self.compute_reward(end_effector_position, collision_detected)

        if collision_detected:
            self.attempts_per_joint += 1
            if self.attempts_per_joint >= self.max_attempts_per_joint:
                self.t += 1
                self.attempts_per_joint = 0

        else:
            self.attempts_per_joint = 0
            self.dh_params.append(self.new_dh_param)  # Append DH parameters
            self.current_position = end_effector_position
            self.visited_positions.append(self.current_position.tolist())
            self.replay_buffer.add(end_effector_position)
            self.render(joint_positions, render)
            self.t += 1
            # print(f"self.visited_positions: {len(self.visited_positions)}")

        if len(self.visited_positions) >= len(self.alpha_comb):
            done = True
            # print("All joints have been visited. Ending the episode.")

        elif self.all_moves_taken(self.current_position):
            done = True
            early_termination = True
            reward = 0 
            reward -= 100
            print("No more valid moves. Ending the episode early.")

        return self.get_state(early_termination), reward, done, {}
    
    def get_state(self, early_termination = False):
        # Total number of joint positions we need (n_joints * 3)
        joint_features_flat = np.zeros(self.n_joints * 3) # Predefine a zero vector

        # Get the actual joint positions from the transformation function
        self.current_jp = self.transformation(self.dh_params)
        current_joint_positions = self.transformation(self.dh_params).flatten()  # Shape: (t * 3,), only for t joints
        if len(current_joint_positions) > 0:
            joint_features_flat[:len(current_joint_positions)] = current_joint_positions
    
        # End-effector position
        if len(self.transformation(self.dh_params)) == 0:
            end_effector_position = self.start_position  # Default position if no transformation yet
        else:
            end_effector_position = self.transformation(self.dh_params)[-1].flatten()  # Shape: (3,)

        previous_actions = self.get_previous_actions()
        # Target voxel grid points
        target_points = self.voxel_grid.flatten()  # Shape: (n_voxels * 3,)

        # Action mask
        action_mask = self.get_action_mask()

        # Joints left to move
        joints_left_flat = np.array([self.n_joints - self.t]).flatten()  # Shape: (1,)

        # Add the last collision position (if it exists)
        if len(self.collision_positions) > 0:
            last_collision_position = np.array(self.collision_positions[-1])
        else:
            last_collision_position = np.zeros(3)  # Default to zero vector if no collision has occurred

        if early_termination:
            joint_features_flat[:] = -1

        # Create a dictionary for the state
        state_dict = {
            "joint_features": torch.tensor(joint_features_flat, dtype=torch.float32).unsqueeze(0),  # (1, n_joints * 3)
            "end_effector": torch.tensor(end_effector_position, dtype=torch.float32).unsqueeze(0),  # (1, 3)
            "target_points": torch.tensor(target_points, dtype=torch.float32).unsqueeze(0),  # (1, n_voxels * 3)
            "action_mask": torch.tensor(action_mask, dtype=torch.float32).unsqueeze(0),  # (1, 3)
            "previous_actions": previous_actions,
            "joints_left": torch.tensor(joints_left_flat, dtype=torch.float32).unsqueeze(0),  # (1, 1)
            "last_collision": torch.tensor(last_collision_position, dtype=torch.float32).unsqueeze(0),  # (1, 3)
        }

        for key, value in state_dict.items():
            if value is None:
                raise ValueError(f"State field {key} is None")

        # Concatenate all components into a single flattened array
        flattened_state = np.concatenate([
            joint_features_flat,       # Zero-padded joint features
            end_effector_position,     # End-effector position
            target_points,             # Target voxel points
            action_mask,               # Action mask
            joints_left_flat,           # Number of joints left to move
            last_collision_position
        ])

        # return torch.tensor(flattened_state, dtype=torch.float32).unsqueeze(0)
        return state_dict
    
    def compute_reward(self, current_position, collisions):
        target_positions = self.voxel_grid
        done = False
        reward = 0
        # Reward for reaching target points
        if np.any(np.all(np.isclose(target_positions, current_position, atol=0.1), axis=1)):
            normalized_t = self.t / len(self.alpha_comb)
            # efficiency_bonus = normalized_t ** 2
            efficiency_bonus = np.exp(normalized_t) - 1
            efficiency_bonus /= (np.exp(1) - 1)  # Normalize to range [0, 1]
            reward += 0.15*(1.0 + efficiency_bonus)  # or a value based on importance
        
        # Penalty for collisions
        if collisions:
            reward -= 0.01  # Set to zero or a small positive value
        
        # Reward for efficiently using turns
        # efficiency_bonus = (len(self.alpha_comb) - (len(self.alpha_comb)-self.t)) / len(self.alpha_comb) #(total turns - remaining turns)/total turns
        # reward += 0.1 * efficiency_bonus
        
        #어짜피 모든 step 끝까지 다 갈거임. 그렇다면 마지막 리워드는 굳이 크게 안줘도 될듯. 
        # If all target points are reached
        # if np.all(np.isclose(target_positions, current_position, atol=1e-1)):
        #     reward += 20.  # larger reward for completing the task
        #     done = True
        # if self.t == len(self.alpha_comb):
        #     done = True
    
        return reward, done

    def get_action_mask(self):
        """Returns a mask indicating valid actions at the current state."""
        action_mask = np.ones(3, dtype=np.float32)  # Assume 3 possible actions: 0, 1, 2

        # Check validity of each action
        for action in range(3):
            angle = self.apply_action(action)
            alpha_index = min(len(self.dh_params), len(self.alpha_comb) - 1)
            new_dh_param = {'a': 1, 'alpha': self.alpha_comb[alpha_index], 'd': 0, 'theta': angle}
            joint_positions = self.transformation(self.dh_params + [new_dh_param])
            candidate_position = np.around(joint_positions[-1], 1)

            if self.replay_buffer.has_visited(candidate_position):
                action_mask[action] = 0.0  # Invalid action

        return action_mask
    
    def get_previous_actions(self):
        """Returns the sequence of previous actions."""
        if len(self.previous_actions) == 0:
            return np.array([-1], dtype=np.int64)  # Placeholder for no actions
        else:
            return np.array(self.previous_actions, dtype=np.int64)

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
    
    def render(self, joint_positions, save_gif = False):
        if save_gif:
            # Insert the new row at the first index (index 0)
            joint_positions = np.insert(joint_positions, 0, self.start_position, axis=0)
            """Render the current path of the agent."""
            x_positions = joint_positions[:, 0]
            y_positions = joint_positions[:, 1]
            z_positions = joint_positions[:, 2]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Plot the entire voxel grid of the target object
            voxel_x = self.voxel_grid[:, 0]
            voxel_y = self.voxel_grid[:, 1]
            voxel_z = self.voxel_grid[:, 2]
            ax.scatter(voxel_x, voxel_y, voxel_z, color='gray', alpha=0.5, label='Voxel Grid')

            ax.plot(x_positions, y_positions, z_positions, '-o', label='Joints & Links', color='blue')
            ax.scatter(x_positions[-1], y_positions[-1], z_positions[-1], color='red', s=100, label='End-Effector')
        
            # Now we visualize the candidates after plotting the current path
            self.visualize_candidates(ax, joint_positions[-1])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Path Finding #{self.t}')
            ax.legend()

            # plt.show()
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
        # Iterate over all possible actions (-90, 0, 90 degrees)
        for action in range(3):
            angle = self.apply_action(action)
            alpha_index = min(len(self.dh_params), len(self.alpha_comb) - 1)
            # Create a new DH parameter set for this action
            new_dh_param = {'a': 1, 'alpha': self.alpha_comb[alpha_index], 'd': 0, 'theta': angle}
            # Calculate the resulting transformation matrix
            joint_positions = self.transformation(self.dh_params + [new_dh_param])
            candidate_position = np.around(joint_positions[-1], 1)

            # Check if the candidate is within the voxel grid and not already visited
            if not self.replay_buffer.has_visited(candidate_position):
                candidates.append(candidate_position)

        # If there are no valid candidates, all moves have been taken
        return len(candidates) == 0

    def visualize_candidates(self, ax, end_effector_position):
        candidates = []
        # Iterate over all possible actions (-90, 0, 90 degrees)
        for action in range(3):
            angle = self.apply_action(action)
            alpha_index = min(len(self.visited_positions), len(self.alpha_comb) - 1)
            # Create a new DH parameter set for this action
            new_dh_param = {'a': 1, 'alpha': self.alpha_comb[alpha_index], 'd': 0, 'theta': angle}
            # Calculate the resulting transformation matrix
            joint_positions = self.transformation(self.dh_params + [new_dh_param])
            candidate_position = np.around(joint_positions[-1], 1)

            # Check if the candidate is within the voxel grid and not already visited
            if not self.replay_buffer.has_visited(candidate_position):
                candidates.append(candidate_position)

        candidates = np.array(candidates)

        # Visualize the candidates
        ax.scatter(candidates[:, 0], candidates[:, 1], candidates[:, 2], color='green', s=50, alpha = 1.0,label='Candidate Positions')
        ax.legend()

        # Check if the current end-effector position is on a target voxel
        if np.any(np.all(np.isclose(self.voxel_grid, end_effector_position, atol=0.1), axis=1)):
            print(f"End-effector reached a target voxel at position: {end_effector_position}")

        return candidates

    def create_gif(self, gif_filename):
        """Create a GIF from saved frames."""
        with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
            for frame in self.frames:
                image = imageio.imread(frame)
                writer.append_data(image)

        # Optionally, clear the frames after GIF creation
        self.frames.clear()

class GFNReconContext:
    def __init__(self, env):
        self.num_node_dim = 3  # 3D position of each joint
        self.num_edge_dim = 4  # 4 DH parameters (a, alpha, d, theta) for transformations

        # Action space mappings
        self.action_space_size = env.action_space.n  # 3 possible actions (-90, 0, 90 degrees)

    def has_n(self):
        # Your environment does not need to predict 'n', so return False
        return False
    
    def state_to_tensor(self, state_dict):
        joint_features = state_dict["joint_features"]
        end_effector = state_dict["end_effector"]
        target_points = state_dict["target_points"]
        action_mask = state_dict["action_mask"]
        joints_left = state_dict["joints_left"]
        last_collision = state_dict["last_collision"]
    
        # Combine these features into a single tensor for passing through the model
        state_tensor = torch.cat([joint_features, end_effector, target_points, action_mask, joints_left, last_collision], dim=1)
        
        return state_tensor

    def state_to_action_index(self, state):
        # Assuming we have a way to convert state to action index.
        # This could involve mapping the current joint positions and available actions.
        # Example: map based on joint positions and angles.
        return np.argmax(state['action_mask'])

    def action_index_to_state(self, action_index):
        print(f"Action index received: {action_index}")
        # Apply the action to move the robot arm in the environment
        new_state, _, _, _ = self.env.step(action_index)
        
        # Return the new state
        return new_state
    
    def prepare_encoder_input(self, state):
        # Extract the target voxel positions from the state
        target_points = state["target_points"]  # Shape: [1, n_voxels * 3]
        seq_len_enc = target_points.shape[1] // 3  # Number of target points
        target_points = target_points.view(1, seq_len_enc, 3)  # Reshape to [batch_size, seq_len_enc, 3]
        return target_points.float()
    
    def prepare_decoder_input(self, state):
        # Extract the sequence of end-effector positions from the joint features
        joint_features = state["joint_features"]  # Shape: [1, n_joints * 3]
        previous_actions = state["previous_actions"]
        collision_history = state[""]
        action_mask = state["action_mask"]
        seq_len_dec = (joint_features != 0).sum() // 3  # Number of joints moved so far
        if seq_len_dec == 0:
            # If no joints have moved yet, use a placeholder (e.g., zeros)
            decoder_input = torch.zeros(1, 1, 3)
        else:
            decoder_input = joint_features[:, :seq_len_dec * 3].view(1, seq_len_dec, 3)
        return decoder_input.float()



