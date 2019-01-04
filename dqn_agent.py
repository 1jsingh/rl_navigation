import numpy as np
import random
from collections import namedtuple, deque

from model import Qnetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


UPDATE_EVERY = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self,state_size,action_size,gamma=0.99,lr=5e-4,
                     buffer_size=int(1e5),batch_size=64,tau=1e-3):
        # defining local and target networks
        self.qnet_local = Qnetwork(state_size,action_size).to(device)
        self.qnet_target = Qnetwork(state_size,action_size).to(device)
        
        # set local and target parameters equal to each other
        self.soft_update(tau=1.0)
        
        # experience replay buffer
        self.memory = ReplayBuffer(buffer_size,batch_size)
        
        # defining variables
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        
        self.t_step = 0
        
        # optimizer
        self.optimizer = optim.Adam(self.qnet_local.parameters(),lr=self.lr)
    
    def step(self,state,action,reward,next_state,done):
        """ saves the step info in the memory buffer and perform a learning iteration
        Input : 
            state,action,reward,state,done : non-batched numpy arrays
        
        Output : 
            none
        """
        # add sample to the memory buffer
        self.memory.add(state,action,reward,next_state,done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        
        # use replay buffer to learn if it has enough samples
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
        
    def learn(self,experiences):
        """ perform a learning iteration by using sampled experience batch
        Input : 
            experience : tuple from the memory buffer
            states, actions, rewards, next_states, dones = experiences
            eg : states.shape = [N,state_size]
        Output : 
            none
        """
        states, actions, rewards, next_states, dones = experiences
        
        # set optimizer grdient to zero
        self.optimizer.zero_grad()
        
        # predicted action value
        q_pred = self.qnet_local.forward(states).gather(1,actions)
        
        # target action value
        ## use double DQNs, refer https://arxiv.org/abs/1509.06461
        next_action_local = self.qnet_local.forward(next_states).max(1)[1]
        q_target = rewards + self.gamma*(1-dones)*self.qnet_target.forward(next_states)[range(self.batch_size),next_action_local].detach().unsqueeze(1)
        # print (q_target.shape,next_action_local.shape)
        
        # defining loss
        loss = F.mse_loss(q_pred,q_target)
        
        # running backprop and optimizer step
        loss.backward()
        self.optimizer.step()
        
        # run soft update
        self.soft_update(self.tau)
        
    def act(self,state,eps=0.):
        """ return the local model's predicted action for the given state
        Input : 
            state : [state_size]
        
        Output : 
            action : scalar action as action space is discrete with dim = 1
        """
        state = torch.from_numpy(state).float().to(device) # converts numpy array to torch tensor
        
        self.qnet_local.eval() # put net in test mode
        with torch.no_grad():
            max_action = np.argmax(self.qnet_local(state).cpu().data.numpy())
        self.qnet_local.train() # put net back in train mode
        
        rand_num = np.random.rand() # sample a random number uniformly between 0 and 1
        
        # implementing epsilon greedy policy
        if rand_num < eps:
            return np.random.randint(self.action_size)
        else: 
            return max_action
        
    def soft_update(self,tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        """
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.batch_size = batch_size
        
    def add(self,state,action,reward,next_state,done):
        e = self.experience(state,action,reward,next_state,done)
        self.buffer.append(e)
    
    def sample(self,random=True):
        if random:
            choose = np.random.choice(range(len(self.buffer)),self.batch_size,replace=False)
            experiences = [self.buffer[i] for i in choose]
        else:
            pass
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states,actions,rewards,next_states,dones)
    
    def __len__(self):
        return len(self.buffer)