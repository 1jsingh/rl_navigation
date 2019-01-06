import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnetwork(nn.Module):
    def __init__(self,state_size,action_size):
        # initialising the super class properties
        super(Qnetwork,self).__init__()
        
        # defining layers
        # Dueling networks, refer https://arxiv.org/abs/1511.06581
        
        # common network layers
        self.fc1 = nn.Linear(state_size,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,128)
        
        # Value network layers
        #self.fc3_v = nn.Linear(64,128)
        self.out_v = nn.Linear(128,1)

        # Advantage estimate layers
        #self.fc3_a = nn.Linear(64,128)
        self.out_a = nn.Linear(128,action_size) 
        
    def forward(self,states):

        # common network
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # value network
        #v = F.relu(self.fc3_v(x))
        v = self.out_v(x)

        # advantage network
        #a = F.relu(self.fc3_a(x))
        a = self.out_a(x)

        # refine advantage
        a_ = a - a.mean(dim=1,keepdim=True)

        # combine v and a_
        q = v + a_ 
        return q