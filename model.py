import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnetwork(nn.Module):
    def __init__(self,state_size,action_size):
        # initialising the super class properties
        super(Qnetwork,self).__init__()
        
        # defining layers
        self.fc1 = nn.Linear(state_size,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,128)
        self.out = nn.Linear(128,action_size)
        
    def forward(self,states):
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        out = F.relu(self.out(x))
        return out