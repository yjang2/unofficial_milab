from collections import namedtuple, deque
import numpy as np
import random
import torch
import os
from model_3d import Linear_QNet3D, QTrainer
import matplotlib.pyplot as plt
from snake_3d import SnakeGame3D

Point3D = namedtuple('Point3D', 'x y z')

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Direction3D:
    FORWARD = 1   # +x
    BACKWARD = 2  # -x
    RIGHT = 3     # +y
    LEFT = 4      # -y
    UP = 5        # +z
    DOWN = 6      # -z

BLOCK_SIZE = 20
class Agent3D:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = Linear_QNet3D(18, 256, 6)
        self. trainer = QTrainer (self.model, lr = LR, gamma = self.gamma)

    def get_state(self, game):
        head = game.head  # snake's head, a Point3D
        
        # Choose the closest food as reference (if any exist)
        if game.food:
            closest_food = min(game.food, key=lambda food: manhattan_distance_3d(food, head))
        else:
            closest_food = head  # fallback: use head itself (won't be used in practice)
        
        # Danger signals: check collisions one block ahead in 6 directions
        danger_front = game.is_collision(Point3D(head.x + BLOCK_SIZE, head.y, head.z))
        danger_back  = game.is_collision(Point3D(head.x - BLOCK_SIZE, head.y, head.z))
        danger_right = game.is_collision(Point3D(head.x, head.y + BLOCK_SIZE, head.z))
        danger_left  = game.is_collision(Point3D(head.x, head.y - BLOCK_SIZE, head.z))
        danger_up    = game.is_collision(Point3D(head.x, head.y, head.z + BLOCK_SIZE))
        danger_down  = game.is_collision(Point3D(head.x, head.y, head.z - BLOCK_SIZE))
        
        # Current movement direction (one-hot for 6 directions)
        dir_front = (game.direction == Direction3D.FORWARD)
        dir_back  = (game.direction == Direction3D.BACKWARD)
        dir_right = (game.direction == Direction3D.RIGHT)
        dir_left  = (game.direction == Direction3D.LEFT)
        dir_up    = (game.direction == Direction3D.UP)
        dir_down  = (game.direction == Direction3D.DOWN)
        
        # Food centroid direction
        # if game.food:
        #     cx = sum(f.x for f in game.food) / len(game.food)
        #     cy = sum(f.y for f in game.food) / len(game.food)
        #     cz = sum(f.z for f in game.food) / len(game.food)
        # else:
        #     cx, cy, cz = head.x, head.y, head.z

        # food_x_left  = cx < head.x
        # food_x_right = cx > head.x
        # food_y_down  = cy < head.y
        # food_y_up    = cy > head.y
        # food_z_down  = cz < head.z
        # food_z_up    = cz > head.z

        food_x_left  = closest_food.x < head.x
        food_x_right = closest_food.x > head.x
        food_y_down  = closest_food.y < head.y
        food_y_up    = closest_food.y > head.y
        food_z_down  = closest_food.z < head.z
        food_z_up    = closest_food.z > head.z

        # Food position relative to head (using the chosen reference food)
        # food_x_left = any(f.x < head.x for f in game.food)
        # food_x_right = any(f.x > head.x for f in game.food)
        # food_y_down = any(f.y < head.y for f in game.food)
        # food_y_up = any(f.y > head.y for f in game.food)
        # food_z_down = any(f.z < head.z for f in game.food)
        # food_z_up = any(f.z > head.z for f in game.food)

        
        state = [
            int(danger_front), int(danger_back), int(danger_right),
            int(danger_left), int(danger_up), int(danger_down),
            int(dir_front), int(dir_back), int(dir_right),
            int(dir_left), int(dir_up), int(dir_down),
            int(food_x_left), int(food_x_right),
            int(food_y_down), int(food_y_up),
            int(food_z_down), int(food_z_up)
        ]
        
        return np.array(state, dtype=int)
        # return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Epsilon-greedy action selection
        self.epsilon = 50 - self.n_games #80이 initial
        final_move = [0] * 6
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 5)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

    def save(self, game_count, record):
        model_folder_path = './model_3d'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, f"model_{game_count}_{record}.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            # Optionally, add replay memory or other parameters here
        }, file_name)

def manhattan_distance_3d(p1, p2):
    return abs(p1.x - p2.x) + abs(p1.y - p2.y) + abs(p1.z - p2.z)

# Visualization function using mplot3d
def visualize_game_3d(game, episode):
    plt.clf()  # clear the current figure
    ax = plt.axes(projection='3d')

    # Extract snake coordinates
    snake_x = [pt.x for pt in game.snake]
    snake_y = [pt.y for pt in game.snake]
    snake_z = [pt.z for pt in game.snake]
    
    # Plot the snake's body as a line with markers
    ax.plot(snake_x, snake_y, snake_z, '-o', color='blue', label='Snake')
    
    # Plot the food
    ax.scatter([food.x for food in game.food],
            [food.y for food in game.food],
            [food.z for food in game.food],
            color='red', s=100, label='Food')

    # Set the limits of the axes based on the grid dimensions
    ax.set_title(f'Ep: {episode} Steps: {game.frame_iteration} Max Steps: {game.food_volume} Score: {game.score}')
    # ax.set_title('testing...')
    ax.set_xlim(0, game.w)
    ax.set_ylim(0, game.h)
    ax.set_zlim(0, game.d)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    
    plt.draw()
    plt.pause(0.001)

def train():
    agent = Agent3D()
    game = SnakeGame3D()
    record = 0
    total_score = 0
    plot_scores = []
    # Turn on interactive mode for matplotlib
    plt.ion()

    direction_mapping = {
        0: Direction3D.FORWARD,
        1: Direction3D.BACKWARD,
        2: Direction3D.RIGHT,
        3: Direction3D.LEFT,
        4: Direction3D.UP,
        5: Direction3D.DOWN
    }
    try:
        while True:
            # Get current state from the environment
            state_old = agent.get_state(game)
            
            # Get action from agent (one-hot vector), then convert to index and map to a 3D direction
            final_move = agent.get_action(state_old)
            action_index = np.argmax(final_move)
            direction = direction_mapping[action_index]
            
            # Execute the action in the environment
            reward, done, score = game.play_step(direction)
            
            # Get the new state after the action
            state_new = agent.get_state(game)
            
            # Train on the most recent step (short-term memory)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            
            # Remember this experience for replay
            agent.remember(state_old, final_move, reward, state_new, done)
            
            # Update visualization using mplot3d
            visualize_game_3d(game, agent.n_games)
            
            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()
                
                if score > record:
                    record = score
                    agent.save(agent.n_games, record)  # Save checkpoint if record achieved
                    
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                # print(f'Game: {agent.n_games} Score: {score} Record: {record} Mean Score: {mean_score:.2f}')
                # Optionally, pause to let you see the final state of this game
                plt.pause(0.5)
                
        # Turn off interactive mode when done (unreachable in an infinite loop)
        plt.ioff()
        plt.show()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Closing plots...")
        plt.close('all')

def retrain():
    agent = Agent3D()
    game = SnakeGame3D()
    checkpoint = torch.load("model_3d/20250409/model_7244_34.pth")
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.model.eval()

    train()

if __name__ == '__main__':
    plt.ion()
    retrain()
#     train()
