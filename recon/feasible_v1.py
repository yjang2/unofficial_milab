import argparse
import collections
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import re
import pyvista as pv

from copy import copy
from utils import *


class FeasibleEnv():
    def __init__(self, env_config):
        self.num_joints = env_config['num_joints']
        self.urdf_path = env_config["urdf_path"]
        self.dir_name = env_config["dir_name"]
        self.target_path = env_config["target_path"]
        self.objfile = self._calling_obj(env_config['target_path'], self.dir_name, index=0)
        self.index = 1
        self.j2j_len = 0.01
        
        # self.reset()
        fig = plt.figure()
        
    def reset(self):
        # self.points_np, self.points_dict = self._make_points()
        self.points_np, self.points_dict = self._load_obj()
        self._check_valid_points(self.points_np)
        self.t = 0
        self.state = dict()
        # self.state["avail_idx"] = [i for i in range(self.n_pts ** 3)]
        self.state["avail_idx"] = self.points_dict.keys()
        self.state["unavail_idx"] = []
        self.state["avail_pts"] = np.array([self.points_np[i] for i in self.state["avail_idx"]])
        self.state["all_pts"] = self.points_np
        
        return self.state
        
    def step(self, point_i):
        self.t += 1
        state = dict()
        reward = 0 # TODO: implement reward
        
        if self.t == 1:
            self.cur_i = point_i
            cur_pt = self.points_dict.pop(point_i)
            self.state["unavail_idx"].append(point_i)
            # self.state["avail_idx"] = [i for i in self.points_dict.keys() if np.linalg.norm(selected_pt-self.points_dict[i]) <= self.args.j2j_len and angle_between(selected_pt, self.points_dict[i]) > np.pi/2]
            self.state["avail_idx"] = [i for i in self.points_dict.keys() if np.linalg.norm(cur_pt-self.points_dict[i]) <= self.j2j_len]
            self.state["avail_pts"] = np.array([self.points_np[i] for i in self.state["avail_idx"]])
            self.state["all_pts"] = self.points_np
        else:
            self.prev_i = self.cur_i
            self.cur_i = point_i
            cur_pt = self.points_dict.pop(point_i)
            prev_pt = self.points_np[self.prev_i]
            self.state["unavail_idx"].append(point_i)
            self.state["avail_idx"] = [i for i in self.points_dict.keys() if np.linalg.norm(cur_pt-self.points_dict[i]) <= self.j2j_len and angle_between(cur_pt-prev_pt, cur_pt-self.points_dict[i]) >= np.pi/2]
            self.state["avail_pts"] = np.array([self.points_np[i] for i in self.state["avail_idx"]])
            self.state["all_pts"] = self.points_np
        done = True if len(self.state["avail_idx"]) == 0 else False

        # state = dict()

        return self.state, reward, done
    
    def render(self):
        directory_path = '/home/milab/Desktop/linux_foldableRobot/render_images/'
        unavail_pts = np.array([self.points_np[i] for i in self.state["unavail_idx"]])
        ax = plt.axes(projection='3d')
    
        if len(self.state["avail_pts"]) > 0:
            avail_x = self.state["avail_pts"][:, 0]
            avail_y = self.state["avail_pts"][:, 1]
            avail_z = self.state["avail_pts"][:, 2]
            ax.scatter(avail_x, avail_y, avail_z, c='blue')
        if len(unavail_pts) > 0:
            unavail_x = unavail_pts[:, 0]
            unavail_y = unavail_pts[:, 1]
            unavail_z = unavail_pts[:, 2]
            ax.scatter(unavail_x, unavail_y, unavail_z, c='red')
            ax.plot(unavail_x, unavail_y, unavail_z, c='red')
        
        ax.set_xlim(-0.1, 0.1)
        ax.set_ylim(-0.1, 0.1)
        ax.set_zlim(-0.1, 0.1)
        
        # Check if the directory exists
        if not os.path.exists(directory_path):
            # If it does not exist, create it
            os.makedirs(directory_path)
            print("Directory created:", directory_path)
        else:
            print("Directory already exists:", directory_path)

        plt.savefig(f"render_images/step_{self.t}")
        plt.clf()
        
    
    def _check_valid_points(self, points):
        assert isinstance(points, np.ndarray), "point cloud points must be numpy array."
        assert len(points.shape) == 2 and points.shape[1] == 3, "point cloud points array shape must be (n, 3)"
        return True
    
    # def _make_points(self):
    #     points_np = np.random.rand(self.n_pts**3, 3)
    #     points_dict = collections.defaultdict(np.array)
    #     i = 0
    #     for x in range(self.n_pts):
    #         for y in range(self.n_pts):
    #             for z in range(self.n_pts):
    #                 points_np[i] = np.array([x, y, z]) / (self.n_pts-1)
    #                 points_dict[i] = np.array([x, y, z]) / (self.n_pts-1)
    #                 i += 1
        
    #     return points_np, points_dict
    
    def _calling_obj(self, path, dirname, index=0):
        all_files = []
        target_dir = os.path.join(path, dirname)
        if not os.path.isdir(target_dir):
            raise ValueError(f"The directory {target_dir} does not exist.")
        for file in sorted(os.listdir(target_dir), key = lambda x: int(re.search(r'(\d+)', x).group() if re.search(r'(\d+)', x) else 0)):
            if file.endswith('.obj'):
                all_files.append(os.path.join(target_dir, file))
        if index < 0 or index >= len(all_files):
            raise ValueError(f"Index out of range. There are {len(all_files)} .obj files but index {self.index} was provided.")
        
        return all_files
    
    def _load_obj(self):
        print("file:",self.objfile[self.index])

        if self.index < 0 or self.index >= len(self.objfile):
            raise ValueError(f"Index out of range. There are {len(self.objfile)} .obj files but index {self.index} was provided.")
        self.mesh = pv.read(self.objfile[self.index])
        scaled_mesh = self.mesh.copy()
        scaled_mesh = scaled_mesh.rotate_x(90)


        # sampled_points = scaled_mesh.sample_points_uniform(n=len(scaled_mesh.faces)*10)
        try:
            voxel_mesh = pv.voxelize(scaled_mesh, density=self.j2j_len)
        except Exception as e:
            print(f"An error occurred during voxelization: {e}")
            return None

        # Convert the voxel centers to a numpy array if needed or directly return
        voxel_centers = voxel_mesh.cell_centers()
        points = np.array(voxel_centers.points)

        points_dict = collections.defaultdict(np.array)
        for i, point in enumerate(points):
            points_dict[i] = point
        
        num_voxels = voxel_mesh.number_of_cells
        print(f"num_voxels: {num_voxels}")

        return points, points_dict


if __name__ == "__main__":
    import random
    
    n_joints = 62

    env_config = {
        'num_joints': n_joints,
        'target_path': "/home/milab/Desktop/linux_foldableRobot/Manifold",
        'dir_name': "bus",
        # "base_path": "/home/milab/Desktop/linux_foldableRobot",
        "urdf_path": "/home/milab/Desktop/linux_foldableRobot/urdf/recon_robot_j62/urdf/recon_robot_j62.urdf",
        "axis": [0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,0,1,0],
    }

    env = FeasibleEnv(env_config)
    state = env.reset()
    env.render()
    done = False
    while not done:
        state, r, d = env.step(random.choice(list(state["avail_idx"])))
        env.render()
        if d:
            break
    
#     # print(state["avail_idx"])
#     # print(state["avail_pts"])
#     # print(state["all_pts"])
    
#     print()
#     state = env.step(10)
#     env.render()
#     print(state["avail_idx"])
#     print(state["avail_pts"])
#     # print(state["all_pts"])
    
#     print()
#     state = env.step(11)
#     env.render()
#     print(state["avail_idx"])
#     print(state["avail_pts"]) 
    
