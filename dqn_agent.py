import numpy as np
import random
from collections import namedtuple, deque

from model import Qnetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


UPDATE_EVERY = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self,state_size,action_size,num_agents,gamma=0.99,lr=1e-3,
                     buffer_size=int(1e6),batch_size=128,tau=1e-3):
        # defining local and target networks
        self.qnet_local = Qnetwork(state_size,action_size).to(device)
        self.qnet_target = Qnetwork(state_size,action_size).to(device)
        
        # set local and target parameters equal to each other
        self.soft_update(tau=1.0)
        
        # experience replay buffer
        self.memory = ReplayBuffer(buffer_size,batch_size)
        
        # number of parallel agents
        self.num_agents = num_agents

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
        for i in range(self.num_agents):
            self.memory.add(state[i],action[i],reward[i],next_state[i],done[i])
        
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
        #states, actions, rewards, next_states, dones,wj,choose = experiences
        states, actions, rewards, next_states, dones = experiences

        # set optimizer grdient to zero
        self.optimizer.zero_grad()
        
        # predicted action value
        q_pred = self.qnet_local.forward(states).gather(1,actions)
        
        # target action value
        ## use double DQNs, refer https://arxiv.org/abs/1509.06461
        next_action_local = self.qnet_local.forward(next_states).max(1)[1]
        q_target = rewards + self.gamma*(1-dones)*self.qnet_target.forward(next_states)[range(self.batch_size),next_action_local].unsqueeze(1)
        
        # compute td error
        td_error = q_target-q_pred
        # update td error in Replay buffer
        # self.memory.update_td_error(choose,td_error.detach().cpu().numpy().squeeze())

        # defining loss
        #loss = ((wj*td_error)**2).mean()
        loss = (td_error**2).mean()
        
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
            max_actions = np.argmax(self.qnet_local(state).cpu().data.numpy(),axis=1)
        self.qnet_local.train() # put net back in train mode
        
        rand_num = np.random.rand(self.num_agents) # sample a random number uniformly between 0 and 1
        rand_actions = np.random.randint(low=0,high=self.action_size,size=self.num_agents)

        check = rand_num < eps
        action = check*rand_actions + np.invert(check)*max_actions

        return action
        
    def soft_update(self,tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        """
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done","td_error"])
        self.batch_size = batch_size
        self.epsilon = 1e-6
        self.alpha = 2
        self.beta = .3

    def add(self,state,action,reward,next_state,done):
        max_td_error = max([e.td_error for e in self.buffer if e is not None]+[0])
        e = self.experience(state,action,reward,next_state,done,max_td_error)
        self.buffer.append(e)
    
    def update_td_error(self,choose,td_errors):
        abs_td_errors = np.abs(td_errors)
        for j,td_error in zip(choose,abs_td_errors):
            self.buffer[j] = self.buffer[j]._replace(td_error=td_error)

    def sample(self,random=False):
        if random:
            choose = np.random.choice(range(len(self.buffer)),self.batch_size,replace=False)
            experiences = [self.buffer[i] for i in choose]
        else:
            # prioritised experience replay
            pi = np.array([e.td_error for e in self.buffer if e is not None]) + self.epsilon
            Pi = pi**self.alpha
            Pi = Pi/np.sum(Pi)
            wi = (len(self.buffer)*Pi)**(-self.beta)
            wi_ = wi/np.max(wi)
            choose = np.random.choice(range(len(self.buffer)),self.batch_size,replace=False,p=Pi)
            experiences = [self.buffer[j] for j in choose]
            wj = [wi_[j] for j in choose]
            wj = torch.from_numpy(np.vstack(wj)).float().to(device)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        

        return (states,actions,rewards,next_states,dones)#,wj,choose)
    
    def __len__(self):
        return len(self.buffer)