import numpy as np
import matplotlib.pyplot as plt
import random
from utils import dh_transform

import os
import re
import trimesh
import numpy as np
import matplotlib.pyplot as plt

class ObjectLoader:
    def __init__(self, path, dirname):
        self.path = path
        self.dirname = dirname
        self.obj_path = None
        self.voxel_grid = None

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
    
    def render(self, joint_positions):
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
        print(voxel_x)
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

# Example usage:
target_path = "/home/milab/Desktop/linux_foldableRobot/Manifold"
dirname = "car"

loader = ObjectLoader(target_path, dirname)
loader._calling_obj(4)
voxels = loader.obj_to_voxels(62)
norm_pos = loader.get_voxel_info()

print(f"Normalized Position numbers: {len(norm_pos)}")

loader.render(norm_pos)
print(np.min(norm_pos, axis=0))


class ReplayBuffer:
    def __init__(self):
        self.visited_positions = []

    def add(self, position):
        self.visited_positions.append(position.tolist())

    def has_visited(self, position):
        return any(np.allclose(position, pos) for pos in self.visited_positions)

class GFNrecon:
    def __init__(self, config):
        super().__init__()
        self.t = 0
        # Initialize ObjectLoader with config
        self.loader = ObjectLoader(config['target_path'], config['dirname'])
        self.loader._calling_obj(config.get('obj_index', 0))  # Load specified .obj file
        self.loader.obj_to_voxels(config.get('target_voxel_count', 62))
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
        self.visited_positions = [self.current_position.tolist()]
        self.replay_buffer.add(self.current_position)

    def new(self):
        self.t = 0
        self.dh_params = []
        self.current_position = self.start_position.copy()
        self.visited_positions = [self.current_position.tolist()]
        self.replay_buffer = ReplayBuffer()  # Clear the buffer
        self.replay_buffer.add(self.current_position)
        return self.current_position

    def step(self, action):
        angle = self.apply_action(action)
        print(f"angle: {angle}")
        new_dh_param = {'a': 1, 'alpha': self.alpha_comb[len(self.visited_positions) - 1], 'd': 0, 'theta': angle}
        self.dh_params.append(new_dh_param)
        joint_positions = self.transformation(self.dh_params, self.start_position)
        end_effector_position = np.around(joint_positions[-1], 1)

        if self.replay_buffer.has_visited(end_effector_position):
            print(f"Collision detected at position {end_effector_position}.")
            self.dh_params.pop()  # Remove the last added DH parameter if there's a collision
        else:
            self.current_position = end_effector_position
            self.visited_positions.append(self.current_position.tolist())
            self.replay_buffer.add(end_effector_position)
            self.render(joint_positions)
            self.t +=1
            
        return self.current_position

    def apply_action(self, action):
        """Apply an action that rotates the joint to a specific angle."""
        angles = {
            0: -90.0,  # Rotate -90 degrees
            1: 0.0,    # No rotation
            2: 90.0,   # Rotate +90 degrees
        }
        return angles.get(action, 0.0)

    def transformation(self, dh_params, start_position):
        T_0_7 = np.eye(4)
        T_0_7[:3, 3] = start_position

        joint_positions = [T_0_7[:3, 3].tolist()]

        for params in dh_params:
            A_i = dh_transform(params['a'], params['alpha'], params['d'], params['theta'])
            T_0_7 = np.dot(T_0_7, A_i)
            joint_positions.append(T_0_7[:3, 3].tolist())

        joint_positions = np.array(joint_positions)
        return joint_positions

    def render(self, joint_positions):
        """Render the current path of the agent."""
        x_positions = joint_positions[:, 0]
        y_positions = joint_positions[:, 1]
        z_positions = joint_positions[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the en
        # tire voxel grid of the target object
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

        plt.show()

    def visualize_candidates(self, ax, end_effector_position):
        candidates = []

        # Iterate over all possible actions (-90, 0, 90 degrees)
        for action in range(3):
            angle = self.apply_action(action)
            # Create a new DH parameter set for this action
            new_dh_param = {'a': 1, 'alpha': self.alpha_comb[len(self.visited_positions) - 1], 'd': 0, 'theta': angle}
            # Calculate the resulting transformation matrix
            joint_positions = self.transformation(self.dh_params + [new_dh_param], self.start_position)
            candidate_position = np.around(joint_positions[-1], 1)

            # Check if the candidate is within the voxel grid and not already visited
            if not self.replay_buffer.has_visited(candidate_position):
                candidates.append(candidate_position)

        candidates = np.array(candidates)

        # Visualize the candidates
        ax.scatter(candidates[:, 0], candidates[:, 1], candidates[:, 2], color='green', s=50, label='Candidate Positions')
        ax.legend()

        # Check if the current end-effector position is on a target voxel
        if np.any(np.all(np.isclose(self.voxel_grid, end_effector_position, atol=1e-1), axis=1)):
            print(f"End-effector reached a target voxel at position: {end_effector_position}")

        return candidates
    
# Example usage:
config = {
    'target_path': "/home/milab/Desktop/linux_foldableRobot/Manifold",
    'dirname': "cuboid",
    'obj_index': 0,  # Load the first .obj file
    'target_voxel_count': 62,  # Target voxel count
}

gfn_env = GFNrecon(config)
gfn_env.new()

for i in range(len(gfn_env.alpha_comb)):
    action = random.randrange(0,3)
    gfn_env.step(action)
