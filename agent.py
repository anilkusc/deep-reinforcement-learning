import torch
import numpy as np
import random
from collections import deque 
import copy

class Agent():
    def __init__(self,input=32,output=4,learning_rate=0.001,gamma = 0.99,epsilon=1.0,replay_memory=10000,memory_batch_size=1000,sync_freq=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input = input
        self.output = output
        self.sync_freq = sync_freq
        self.sync_counter = 0
        self.memory_batch_size = memory_batch_size
        self.replay_memory = deque(maxlen=replay_memory)
        self.model = torch.nn.Sequential(
        torch.nn.Linear(self.input, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, self.output)
    ).to(self.device)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon

    def train_backpropagation(self,X,Y):
        X, Y = X.to(self.device), Y.to(self.device)
        #Y is real reward value
        #X is predicted reward value
        self.optimizer.zero_grad()
        loss = self.loss_fn(X,Y)
        loss.backward()
        self.optimizer.step()
        return loss

    def state_to_tensor_input_encoding(self,state):
        normalized_state = state / 255.0
        return torch.tensor(normalized_state, dtype=torch.float32).to(self.device)

    def state_to_tensor_input(self,state):
        #normalized_state = state / float(input_size-1)
        nn_input = np.zeros(self.input)
        nn_input[state] = 1.0
        pytorch_input = torch.from_numpy(nn_input).float().to(self.device)
        return pytorch_input
    
    def preprocess_image(self,state):
        normalized = state.astype(np.float32) / 255.0
        flattened = normalized.flatten()
        return torch.from_numpy(flattened).float().to(self.device)
    
    def QPolicy(self,qval):
        qval_ = qval.cpu().data.numpy() 
        if (random.random()< self.epsilon):
            action= np.random.randint(0,self.output)
        else:
            action= np.argmax(qval_)
        return action
    
    def calculate_reward(self,reward,next_state,done):

        with torch.no_grad():
            newQ = self.target_model(next_state)
        maxQ = torch.max(newQ, dim=1)[0]
        # done 0 veya 1 olarak gelir. 0 ise etki etmez 1 ise Y=reward olur
        Y = reward + (self.gamma * maxQ * (1 - done))
        #Y = torch.Tensor([Y]).to(self.device).detach() 
        Y = Y.to(self.device).detach() 
        return Y
    
    def save(self):
        torch.save(self.model.state_dict(), "model.pth")

    def load(self):
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
    
    def update(self):
        loss = 0
        if len(self.replay_memory) > self.memory_batch_size:
            minibatch = random.sample(self.replay_memory,self.memory_batch_size)
            state_batch = torch.stack([s1 for (s1, a, r, s2, d) in minibatch]).to(self.device)
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(self.device)
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(self.device)
            next_state_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).to(self.device)
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(self.device)

            Y = self.calculate_reward(reward_batch,next_state_batch,done_batch)
            
            qval = self.model(state_batch)
            X = qval.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = self.train_backpropagation(X,Y)
        if self.sync_counter >= self.sync_freq:
            self.target_model.load_state_dict(self.model.state_dict())
            self.sync_counter = 0
        self.sync_counter += 1
        return loss