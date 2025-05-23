from collections import namedtuple, deque
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange
import random

BLOCK_SIZE = 10
MAX_MEMORY = 100_000

Point3D = namedtuple('Point3D', 'x y z')

# ---------- Robot with DH-based θ Actions ----------
def standard_dh_transform(a, alpha, d, theta):
    theta = math.radians(theta)
    alpha = math.radians(alpha)
    return np.array([
        [math.cos(theta), -math.sin(theta) * math.cos(alpha),  math.sin(theta) * math.sin(alpha), a * math.cos(theta)],
        [math.sin(theta),  math.cos(theta) * math.cos(alpha), -math.cos(theta) * math.sin(alpha), a * math.sin(theta)],
        [0,                math.sin(alpha),                  math.cos(alpha),                 d],
        [0,                0,                                0,                               1]
    ])

class HybridRobot:
    def __init__(self, start_pos=np.array([0,0,0]), block_size=BLOCK_SIZE):
        self.block_size = block_size
        self.start_pos = np.array(start_pos)
        self.alpha_comb_list = [0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0,
                           0,0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0,
                           0,0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0,
                           0,0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0]   
        self.alpha_comb = [-90 if x == 1 else 0 for x in self.alpha_comb_list]
        self.memory = deque(maxlen = MAX_MEMORY)
        self.reset()

    def reset(self):
        self.dh_params = []
        self.head = self.start_pos.copy()
        self.current_joint = 0
        self.action_history = []

    def apply_action(self, action_idx):
        angle = {0: -90.0, 1: 0.0, 2: 90.0}[action_idx]
        alpha = self.alpha_comb[self.current_joint]
        new_param = {'a': self.block_size, 'alpha': alpha, 'd': 0, 'theta': angle}
        self.dh_params.append(new_param)
        self.current_joint += 1
        self.action_history.append(angle)
        T = self.transformation()
        self.head = T[-1]
        return self.head, angle

    def transformation(self):
        T = np.eye(4)
        T[:3, 3] = self.start_pos
        joints = [T[:3, 3].copy()]
        for p in self.dh_params:
            T = T @ standard_dh_transform(p['a'], p['alpha'], p['d'], p['theta'])
            joints.append(T[:3, 3].copy())
        return np.array(joints[1:])

class HybridGame:
    def __init__(self):
        self.w, self.h, self.d = 640, 480, 320
        self.robot = HybridRobot(start_pos=np.array([320, 240, 160]))
        self.block_size = BLOCK_SIZE
        self.max_joints = len(self.robot.alpha_comb)
        self.reset()

    def reset(self):
        self.robot.reset()
        self.score = 0
        self.step_iter = 0
        self.snake = [self.robot.head.copy()]
        self.place_food()
        return self.get_state()

    def place_food(self, num=40, offset_range=2):
        head = self.robot.head
        self.food = []
        while len(self.food) < num:
            dx = random.randint(-offset_range, offset_range) * self.block_size
            dy = random.randint(-offset_range, offset_range) * self.block_size
            dz = random.randint(-offset_range, offset_range) * self.block_size
            x = max(0, min(head[0] + dx, self.w - self.block_size))
            y = max(0, min(head[1] + dy, self.h - self.block_size))
            z = max(0, min(head[2] + dz, self.d - self.block_size))
            pt = np.array([x, y, z])
            if not any(np.array_equal(pt, f) for f in self.food):
                self.food.append(pt)

    def is_collision(self, pt):
        out_of_bounds = (
            pt[0] < 0 or pt[0] >= self.w or
            pt[1] < 0 or pt[1] >= self.h or
            pt[2] < 0 or pt[2] >= self.d
        )
        self_collide = any(np.allclose(pt, s) for s in self.snake[1:])
        return out_of_bounds or self_collide

    def play_step(self, action_idx):
        self.step_iter += 1
        # head, theta = self.robot.apply_action(action)

        head, theta = self.robot.apply_action(action_idx)

        self.snake.insert(0, head.copy())

        reward = 0.0
        done = False

        # if self.is_collision(head) or self.step_iter > self.max_joints:
        if self.is_collision(head):
            print("Collision checked!")
            done = True
            return -10.0, done, self.score

        for i, f in enumerate(self.food):
            if np.linalg.norm(f - head) < self.block_size / 2:
                self.food.pop(i)
                reward += 10.0
                self.score += 1
                break

        return reward, done, self.score

    def get_state_old(self):
        if self.food:
            closest_food = min(self.food, key=lambda f: np.linalg.norm(f - self.robot.head))
        else:
            closest_food = self.robot.head

        joint_pos = self.robot.transformation()
        if joint_pos.ndim == 2 and joint_pos.shape[0] > 0:
            z_axis = np.array([0, 0, 1])  # placeholder
            R_mat = np.eye(3)  # fallback rotation
            food_vec_local = closest_food - self.robot.head
        else:
            z_axis = np.array([0, 0, 1])
            R_mat = np.eye(3)
            food_vec_local = closest_food - self.robot.head

        dist = np.linalg.norm(food_vec_local)
        food_dir = food_vec_local / (dist + 1e-5)

        alpha_flag = self.robot.alpha_comb[len(self.robot.dh_params)] if len(self.robot.dh_params) < len(self.robot.alpha_comb) else 0
        joint_norm = len(self.robot.dh_params) / self.max_joints

        state = np.concatenate([food_vec_local / 1000.0, z_axis, [alpha_flag], [joint_norm]])
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def get_state(self):
        head = self.robot.head  # snake's head, a Point3D

        #choose the closest food as ref (if any exist)
        if self.food:
            closest_food = min(self.food, key=lambda food: self.manhattan_distance_3d(food, head))
        else:
            closest_food = head  # fallback: use head itself (won't be used in practice)

        danger_front = self.is_collision(Point3D(head[0] + BLOCK_SIZE, head[1], head[2]))
        danger_back  = self.is_collision(Point3D(head[0] - BLOCK_SIZE, head[1], head[2]))
        danger_right = self.is_collision(Point3D(head[0], head[1] + BLOCK_SIZE, head[2]))
        danger_left  = self.is_collision(Point3D(head[0], head[1] - BLOCK_SIZE, head[2]))
        danger_up    = self.is_collision(Point3D(head[0], head[1], head[2] + BLOCK_SIZE))
        danger_down  = self.is_collision(Point3D(head[0], head[1], head[2] - BLOCK_SIZE))
    
        food_x_left  = closest_food[0] < head[0]
        food_x_right = closest_food[0] > head[0]
        food_y_down  = closest_food[1] < head[1]
        food_y_up    = closest_food[1] > head[1]
        food_z_down  = closest_food[2] < head[2]
        food_z_up    = closest_food[2] > head[2]

        state = [
            int(danger_front), int(danger_back), int(danger_right),
            int(danger_left), int(danger_up), int(danger_down),
            int(food_x_left), int(food_x_right),
            int(food_y_down), int(food_y_up),
            int(food_z_down), int(food_z_up)
        ]
    
        return np.array(state, dtype=int)
        
    def manhattan_distance_3d(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
