import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Linear_QNet3D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet3D, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model_3d.pth'):
        model_folder_path = './model_3d'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.loss = 0.0

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        # 1: predicted Q values with current state
        pred = self.model(state)
        
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.loss = loss
        self.optimizer.step()

class Kinematic_QNet3D(nn.Module):
    def __init__(self, input_size, hidden_size, n_actions):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.head_q = nn.Linear(hidden_size, n_actions)
        self.head_aux = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        q = self.head_q(feat)
        z_pred = self.head_aux(feat)       # 로컬 Z축 예측값
        return q, z_pred

class Kinematic_QTrainer:
    def __init__(self, model, lr, gamma, aux_weight=0.1):
        self.lr = lr; self.gamma = gamma; self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion_q = nn.MSELoss()
        self.criterion_aux = nn.MSELoss()
        self.aux_weight = aux_weight

    def train_step(self, state, action, reward, next_state, done, z_true=None):
        state  = torch.tensor(state,  dtype=torch.float)
        next_s = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action,      dtype=torch.float)
        reward = torch.tensor(reward,      dtype=torch.float)

        if len(state.shape)==1:  # batch 차원 맞추기
            state  = state.unsqueeze(0)
            next_s = next_s.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done   = (done, )
            if z_true is not None:
                z_true = torch.tensor(z_true, dtype=torch.float).unsqueeze(0)

        q_pred, z_pred = self.model(state)
        q_take = torch.sum(q_pred*action, dim=1)

        with torch.no_grad():
            q_next,_ = self.model(next_s)
            q_target = reward + self.gamma * torch.max(q_next, dim=1)[0] * (~torch.tensor(done))
        loss_q = self.criterion_q(q_take, q_target)

        if z_true is not None:
            if isinstance(z_true, np.ndarray):
                z_true = torch.tensor(z_true, dtype=torch.float)
            if len(z_true.shape) == 1:
                z_true = z_true.unsqueeze(0)
            loss_aux = self.criterion_aux(z_pred, z_true)
        else:
            loss_aux = 0

        loss = loss_q + self.aux_weight * loss_aux
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
