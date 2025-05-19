import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SnakeNetAI(nn.Module):
    
    def __init__(self, input_size, ):
        super().__init__()
        self.l1=nn.Linear(input_size,128)
        self.l2=nn.Linear(128,64)
        self.l3=nn.Linear(64,3)
        self.relu = nn.ReLU()
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x =self.relu(self.l1(x))
        x =self.relu(self.l2(x))
        x = self.l3(x)
        return x
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def cr_ld(self, path):
        if not os.path.exists(path):
            print(f"[INFO] Model file not found at '{path}'. Creating a new model...")
            model = self.__class__(self.l1.in_features).to(device)
            torch.save(model.state_dict(), path)
            print(f"[INFO] New model saved to '{path}'")


class SnakeLearningAI:

    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def compute_Q_live(self, experience):
        #state_current, action, reward, state_next, done = [experience]
        if isinstance(experience[0], (list, tuple)):  # batch
            state_current, action, reward, state_next, done = zip(*experience)
        else:  # single sample
            state_current, action, reward, state_next, done = experience

        state_current = torch.tensor(np.array(state_current),dtype=torch.float).cuda()
        state_next = torch.tensor(np.array(state_next),dtype=torch.float).cuda()
        action = torch.tensor(action,dtype=torch.long).cuda()
        reward = torch.tensor(reward,dtype=torch.float).cuda()
        
        if (len(state_current.shape) == 1):
            state_current = torch.unsqueeze(state_current,0)
            state_next = torch.unsqueeze(state_next,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            done = (done, )

        pred = self.model(state_current) # gives the Q-values for all possible values
        target = pred.clone().cuda()
        for idx in range (len(done)):
            if done[idx]:
                q_value=reward[idx]
            else:
                q_value=reward[idx] + torch.max(self.target_model(state_next[idx])) # .max returns the max value in the tensor
            target[idx][torch.argmax(action[idx]).item()] = q_value   # .argmax returns the Index of the max value in the tensor
        self.optim.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optim.step()
        
        return loss.item()
    
    
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.lr = lr
        self.optim = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        #Target network
        self.target_model = SnakeNetAI(model.l1.in_features).to(device)
        self.target_model.load_state_dict(model.state_dict())  # Copy weights initially
        self.target_model.eval()  # Don't train this network

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        



    
    
    
        

