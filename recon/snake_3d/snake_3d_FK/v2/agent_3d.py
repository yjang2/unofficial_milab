from collections import namedtuple, deque
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model_3d import Linear_QNet3D, QTrainer
import matplotlib.pyplot as plt
from snake_3d import HybridGame
from tqdm import trange
import wandb

Point3D = namedtuple('Point3D', 'x y z')
wandb.init(project="shape_morphing_robot", name="local_frame_training")


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
        self.epsilon = 80 - self.n_games #80이 initial
        # self.epsilon = 0
        if random.random() < (self.epsilon / 200):
            return random.randint(0, 2)       # <-- integer
        else:
            state_v = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_v)
            return torch.argmax(prediction).item()  # <-- integer

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

class LocalAgent(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 3)
        self.aux_food_dir_head = nn.Linear(hidden_dim, 3)
        self.aux_reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        logits = self.action_head(feat)
        pred_food_dir = self.aux_food_dir_head(feat)
        pred_reward = self.aux_reward_head(feat)
        return logits, pred_food_dir, pred_reward
    
class HybridAgent(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(12, 128), nn.ReLU(),
        #     nn.Linear(128, 128), nn.ReLU(),
        #     nn.Linear(128, 3)
        # )
        self.gamma = 0.9
        self.n_games = 0
        self.epsilon = 0

        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = Linear_QNet3D(12, 256, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)
    
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
        self.epsilon = 80 - self.n_games
        if random.random() < self.epsilon/200:
            return random.randint(0,2)
        else:
            with torch.no_grad():
                pred = self.model(torch.tensor(state, dtype=torch.float))
            return int(torch.argmax(pred))

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

def visualize_game_3d(game, episode):
    plt.clf()  # clear the current figure
    ax = plt.axes(projection='3d')

    pts   = [p.flatten() for p in game.snake]
    snake = np.stack(pts)   # now works: all shapes (3,)  
    ax.plot(snake[:, 0], snake[:, 1], snake[:, 2], '-o', color='blue', label='Robot Path')

    if game.food:
        food = np.array(game.food)
        ax.scatter(food[:, 0], food[:, 1], food[:, 2], color='red', s=50, label='Food')

    ax.set_xlim(0, game.w)
    ax.set_ylim(0, game.h)
    ax.set_zlim(0, game.d)
    ax.set_title('Robot Simulation')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend()
    plt.draw()
    plt.pause(0.01)

    return ax

def train_original(agent = None):
    if agent is None:
        agent = HybridAgent()
    game = HybridGame()
    record = 0
    total_score = 0
    plot_scores = []
    # Turn on interactive mode for matplotlib
    plt.ion()

    try:
        while True:
            # Get current state from the environment
            state_old = agent.get_state(game)
            z_true_old = state_old[-4:-1]  # get_state가 마지막에 z_axis 세 개를 넣었다면

            # Get action from agent (one-hot vector), then convert to index and map to a 3D direction
            final_move = agent.get_action(state_old)
            action_index = np.argmax(final_move)
            direction = direction_mapping[action_index]
            
            # Execute the action in the environment
            reward, done, score = game.play_step(direction)
            
            # Get the new state after the action
            state_new = agent.get_state(game)
            
            # Train on the most recent step (short-term memory)
            agent.train_short_memory(state_old, final_move, reward, state_new, done, z_true_old)
            
            # Remember this experience for replay
            agent.remember(state_old, final_move, reward, state_new, done, z_true_old)
            
            # Update visualization using mplot3d
            visualize_game_3d(game, agent.n_games)
            # game.visualize_local_axes()
            
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

def train(num_episode = 2000,load = True):
    game = HybridGame()
    agent = load_model('model_3d/model_701_9.pth') if load else HybridAgent()
    record = 0
    total_score = 0
    # plot_scores = []
    # Turn on interactive mode for matplotlib
    plt.ion()

    try:
        # while True:
        for episode in trange(num_episode, desc="Training Progress"):
            state_old = game.reset()
            for step in range(game.max_joints):
                # Get current state from the environment
                state_old = game.get_state()
                action_index = agent.get_action(state_old)
                print(f"action index {action_index}")

                final_move = [0, 0, 0]
                final_move[action_index] = 1
                
                # Get action from agent (one-hot vector), then convert to index and map to a 3D direction
                reward, done, score = game.play_step(action_index)
                                
                # Get the new state after the action
                state_new = game.get_state()
                
                # Train on the most recent step (short-term memory)
                agent.train_short_memory(state_old, final_move, reward, state_new, done)
                
                # Remember this experience for replay
                agent.remember(state_old, final_move, reward, state_new, done)
                
                # Update visualization using mplot3d
                if step % 5 == 0:
                    visualize_game_3d(game, agent.n_games)
                
                if done:
                    game.reset()
                    agent.n_games += 1
                    agent.train_long_memory()
                    
                    if score > record:
                        record = score
                        agent.save(agent.n_games, record)  # Save checkpoint if record achieved
                        
                    # plot_scores.append(score)
                    total_score += score
                    mean_score = total_score / agent.n_games
                    # print(f'Game: {episode} Score: {score} Record: {record} Mean Score: {mean_score:.2f}')
                    # Optionally, pause to let you see the final state of this game
                    plt.pause(0.5)

                    print(f"Episode {episode}, Score: {score}")

                    wandb.log({
                        "score": score,
                        "mean_score": mean_score,
                        "q_loss": agent.trainer.loss.item(),
                    })
                    break

                
        # Turn off interactive mode when done (unreachable in an infinite loop)
        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
        print("KeyboardInterrupt caught. Closing plots...")
        plt.close('all')


        # if step % 5 == 0:
        #     ax = visualize_game_3d(game, ax=ax, title=f"Ep {ep} | Step {step} | Score: {game.score}")


def retrain():
    game = SnakeGame3D()
    checkpoint = "model_3d/model_1341_168.pth"
    agent = load_model(checkpoint, train_mode=False)
    train(agent)

def load_model(checkpoint_path, agent=None, train_mode=True):
    """
    Loads a saved model checkpoint into the agent.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        agent (Agent3D, optional): An already created agent instance. 
            If not provided, a new Agent3D instance is created.
        train_mode (bool): If True, sets model to training mode, else evaluation mode.
    
    Returns:
        Agent3D: The agent with loaded model and optimizer states.
    """
    # Create a new agent if one is not provided.
    if agent is None:
        agent = HybridAgent()
    
    # Load the checkpoint (make sure the checkpoint file exists at the path)
    checkpoint = torch.load(checkpoint_path)
    
    # Load state dictionaries for both the model and the optimizer
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Set the model mode based on what you plan to do next.
    if train_mode:
        agent.model.train()
    else:
        agent.model.eval()
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    return agent

if __name__ == '__main__':
    # retrain()
    # load_model(checkpoint_path='model_3d/model_701_9.pth')
    train(load=False)
