import random
from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection
import math

# Define a 3D point (grid aligned)
Point3D = namedtuple('Point3D', 'x y z')

# Settings
BLOCK_SIZE = 20
GRID_WIDTH = 640
GRID_HEIGHT = 480
GRID_DEPTH = 320

# Define the 6 cardinal directions for 3D (non-diagonal)
class Direction3D:
    FORWARD = 1   # +x
    BACKWARD = 2  # -x
    RIGHT = 3     # +y
    LEFT = 4      # -y
    UP = 5        # +z
    DOWN = 6      # -z

class SnakeGame3D:
    def __init__(self, w=GRID_WIDTH, h=GRID_HEIGHT, d=GRID_DEPTH):
        self.w = w
        self.h = h
        self.d = d

        self.food_num = 20
        self.food_volume = 0
        self.action_history = []
        self.current_joint = 0
        self.alpha_comb = [0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0,
                           0,0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0,
                           0,0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0,
                           0,0,0,1,0,1,0,1,
                           0,1,0,1,0,1,0,0]              
        self.reset()

    def reset(self):
        # Start in the center of the 3D grid
        self.head = Point3D(self.w // 2, self.h // 2, self.d // 2)
        self.snake = [
            self.head,
            Point3D(self.head.x - BLOCK_SIZE, self.head.y, self.head.z),
            Point3D(self.head.x - 2 * BLOCK_SIZE, self.head.y, self.head.z)
        ]

        self.initial_length = len(self.snake)

        self.score = 0
        self.frame_iteration = 0
        self.step_iter = 0
        self.food_volume = 0
        self.direction = Direction3D.FORWARD
        self.place_food()

    def place_food(self, num_food=10, offset_range=3):
        head = self.head  # current snake head (Point3D)
        food_points = []
        num_food = self.food_num
        
        while len(food_points) < num_food:
            # Choose a random offset in blocks (converted to pixels)
            dx = random.randint(-offset_range, offset_range) * BLOCK_SIZE
            dy = random.randint(-offset_range, offset_range) * BLOCK_SIZE
            dz = random.randint(-offset_range, offset_range) * BLOCK_SIZE
            
            new_x = max(0, min(head.x + dx, self.w - BLOCK_SIZE))
            new_y = max(0, min(head.y + dy, self.h - BLOCK_SIZE))
            new_z = max(0, min(head.z + dz, self.d - BLOCK_SIZE))
            
            candidate = Point3D(new_x, new_y, new_z)
            if candidate in food_points or candidate in self.snake:
                continue
            
            food_points.append(candidate)
        
        # dx = random.randint(-offset_range, offset_range) * BLOCK_SIZE
        # dy = random.randint(-offset_range, offset_range) * BLOCK_SIZE
        # dz = random.randint(-offset_range, offset_range) * BLOCK_SIZE

        # new_x = max(0, min(head.x + dx, self.w - BLOCK_SIZE))
        # new_y = max(0, min(head.y + dy, self.h - BLOCK_SIZE))
        # new_z = max(0, min(head.z + dz, self.d - BLOCK_SIZE))

        # candidate = Point3D(new_x, new_y, new_z)

        # food_points.append(candidate)

        self.food = food_points
        # self.food_volume = len(self.food)*offset_range
        self.food_volume = 62

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Check boundaries in 3D
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h or pt.z < 0 or pt.z >= self.d:
            return True
        # Check self-collision
        if pt in self.snake[1:]:
            return True
        return False

    def move(self, direction):
        # Update head position based on the chosen direction
        x, y, z = self.head
        if direction == Direction3D.FORWARD:
            x += BLOCK_SIZE
        elif direction == Direction3D.BACKWARD:
            x -= BLOCK_SIZE
        elif direction == Direction3D.RIGHT:
            y += BLOCK_SIZE
        elif direction == Direction3D.LEFT:
            y -= BLOCK_SIZE
        elif direction == Direction3D.UP:
            z += BLOCK_SIZE
        elif direction == Direction3D.DOWN:
            z -= BLOCK_SIZE
        self.head = Point3D(x, y, z)

    def play_step(self, direction):
        # self.action_history.append(self.direction)
        self.frame_iteration += 1 #gloabal iteration
        self.step_iter += 1 #local iteration
        self.direction = direction
        self.move(direction)
        self.snake.insert(0, self.head)  # add new head position

        reward = 0
        game_over = False

        if self.is_collision() or self.step_iter > self.food_volume:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # step_penalty = -0.01
        # reward += step_penalty

        if self.head in self.food:
            self.score += 1
            reward = 10
            # self.place_food()
            
            self.food = [f for f in self.food if f!=self.head]

            if len(self.food) == 0:
                self.score += 5
                reward += 40
                self.snake = self.snake[:self.initial_length]
                self.step_iter = 0 
                self.place_food()
        # else:
        #     self.snake.pop()  # remove tail if no food eaten

        return reward, game_over, self.score
    
    def get_dh_params(self):
        dh_params = []
        for i, direction in enumerate(self.action_history):
            alpha = -90 if self.alpha_comb[i] == 1 else 0
            theta = self.direction_to_theta(direction)
            dh_params.append({
                'theta': theta,
                'd': self.link_length,
                'a': 0,
                'alpha': alpha
            })
        return dh_params
    
# Main loop for testing and visualization
# if __name__ == '__main__':
#     plt.ion()  # turn on interactive mode for matplotlib
#     game = SnakeGame3D()
    
#     # Define possible actions for testing
#     possible_actions = [
#         Direction3D.FORWARD, Direction3D.BACKWARD,
#         Direction3D.RIGHT, Direction3D.LEFT,
#         Direction3D.UP, Direction3D.DOWN
#     ]
    
#     running = True
#     while running:
#         action = random.choice(possible_actions)
#         reward, game_over, score = game.play_step(action)
        
#         print(f"Head: {game.head} | Food: {game.food} | Score: {score}")
        
#         if game_over:
#             print("Game Over! Resetting...")
#             game.reset()
        
#         # Handle window close events
#         for event in plt.get_current_fig_manager().canvas.figure.canvas.callbacks.callbacks.get('close_event', {}).values():
#             running = False

#     plt.ioff()
#     plt.show()
