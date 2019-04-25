import numpy as np
import random
from collections import namedtuple, deque

from agents.model import Qnetwork
from agents.bst import FixedSize_BinarySearchTree

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


UPDATE_EVERY = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self,state_size,action_size,num_agents,gamma=0.99,lr=1e-3,
                     buffer_size=int(1e6),batch_size=128,tau=1e-3,random_seed=0):
        # defining local and target networks
        self.qnet_local = Qnetwork(state_size,action_size).to(device)
        self.qnet_target = Qnetwork(state_size,action_size).to(device)
        
        # set local and target parameters equal to each other
        self.soft_update(tau=1.0)
        
        # experience replay buffer
        self.memory = ReplayBuffer(buffer_size,random_seed)
        
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
        
        # initialize time step counter
        self.t_step = 0

        # set random seed
        self.seed = random.seed(random_seed)
        
        # optimizer
        self.optimizer = optim.Adam(self.qnet_local.parameters(),lr=self.lr)
    
    def step(self,state,action,reward,next_state,done):
        """ saves the step info in the memory buffer and perform a learning iteration
        Input : 
            state,action,reward,state,done : non-batched numpy arrays
        
        Output : 
            none
        """

        # get max priority
        max_priority = self.memory._get_max_priority()
        #print ("max priority: {:.2f}".format(max_priority))

        # add sample to the memory buffer
        for i in range(self.num_agents):
            self.memory.add(state[i],action[i],reward[i],next_state[i],done[i],max_priority)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        
        # use replay buffer to learn if it has enough samples
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
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
        
        states,actions,rewards,next_states,dones, idxs, is_weights = experiences

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
        self.memory.update_priorities(idxs,td_error.detach().cpu().numpy().squeeze())

        # compute loss
        loss = ((is_weights*td_error)**2).mean()
        
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
    def __init__(self,buffer_size,seed,alpha=0.4,beta=0.4):
        self.buffer = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.tree = FixedSize_BinarySearchTree(capacity=buffer_size)
        self.epsilon = 1e-5
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = 1e-3
        self.base_priority = self.epsilon**self.alpha

    def add(self,state,action,reward,next_state,done,max_priority):
        self.tree.add(max_priority)    

        e = self.experience(state,action,reward,next_state,done)
        self.buffer.append(e)
    
    def _get_max_priority(self):
        try:
            max_priority = self.tree.max_value()
        except:
            max_priority = self.base_priority

        return max_priority

    def update_priorities(self,idxs,td_errors):
        new_priorities = np.abs(td_errors)**self.alpha

        #print ("update: {:.2f},{:.2f},{:.2f}".format(self.tree.value_sum,np.max(self.tree.values),np.max(new_priorities)))
        for idx,new_priority in zip(idxs,new_priorities):
            self.tree.update(new_priority,idx)

    def sample(self,batch_size):
        sampling_probabilities = np.array(self.tree.values)/self.tree.value_sum
        idxs = np.random.choice(range(self.tree.size),batch_size,replace=False,p=sampling_probabilities)
        sampling_probabilities = sampling_probabilities[idxs]
        experiences = [self.buffer[i] for i in idxs]
        is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        is_weights = torch.from_numpy(np.vstack(is_weights)).float().to(device)

        # increment beta
        self.beta = min(1.0, self.beta+self.beta_increment_per_sampling)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones, idxs, is_weights
    
    def __len__(self):
        return len(self.buffer)