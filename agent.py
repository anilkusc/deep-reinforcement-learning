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
        self.transitions = []
        self.model = torch.nn.Sequential(
        torch.nn.Linear(self.input, 256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, 256),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(256, self.output),
        torch.nn.Softmax(),
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
        #loss = self.loss_fn(X,Y)
        loss = self.loss_fn_preds(X,Y)
        loss.backward()
        self.optimizer.step()
        return loss

    def loss_fn_preds(self,predictions,rewards):
        # We are taking logaritmic because this will penalty more if predictions close to 0.
        # For example if model find probability as very close to 0.(0.01) But it should be more. We are penalty this attidude more with logaritmic function.
        total_rewards = torch.sum(rewards*torch.log(predictions)).to(self.device)
        return total_rewards
        # Need to study more about why we multiply it with -1 :D 
        #return -1 * total_rewards 

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
    
    def Action_Selector(self,state_):
        action_probabilities = self.model(state_)
        action_probabilities_cpu = action_probabilities.cpu()
        actions = np.arange(self.output)
        action = np.random.choice(actions,p = action_probabilities_cpu.data.numpy())
        return action
    
    def discount_rewards(self,rewards):
        # it is going to create [N,N-1,... 3,2,1] array for multiplying it with gamma 
        discount_batch = torch.pow(self.gamma,torch.arange(len(rewards)).float()).to(self.device)
        # If first move most important you do not need to flip. But reward is at the and it is better to flip it because the most important move is at the last.
        #discount_batch_flipped = torch.flip(discount_batch)
        discount_return =  discount_batch * rewards
        #normalize reward
        discount_return /= discount_return.max()
        return discount_return

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
    
    def batch_state_transition(self):
        minibatch = self.transitions    
        reward_batch = torch.Tensor([r for (s,a,r) in minibatch]).to(self.device)
        state_batch = torch.stack([s for (s,a,r) in minibatch]).to(self.device)
        action_batch = torch.Tensor([a for (s,a,r) in minibatch]).to(self.device)
        discounted_rewards = self.discount_rewards(reward_batch)
        prediction_batch = self.model(state_batch)
        probability_batch = prediction_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()
        return probability_batch,discounted_rewards

    def batch_state_replay_memory(self,minibatch):

        state_batch = torch.stack([s1 for (s1, a, r, s2, d) in minibatch]).to(self.device)
        action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch]).to(self.device)
        reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch]).to(self.device)
        next_state_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch]).to(self.device)
        done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch]).to(self.device)
        return state_batch,action_batch,reward_batch,next_state_batch,done_batch

    def update(self):
        loss = 0
        if len(self.replay_memory) > self.memory_batch_size:
            minibatch = random.sample(self.replay_memory,self.memory_batch_size)
            state_batch,action_batch,reward_batch,next_state_batch,done_batch = self.batch_state_replay_memory(minibatch)

            Y = self.calculate_reward(reward_batch,next_state_batch,done_batch)
            
            qval = self.model(state_batch)
            X = qval.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = self.train_backpropagation(X,Y)
        if self.sync_counter >= self.sync_freq:
            self.target_model.load_state_dict(self.model.state_dict())
            self.sync_counter = 0
        self.sync_counter += 1
        return loss

    def REINFORCE(self):
        probability_batch,discounted_rewards = self.batch_state_transition()
        return self.train_backpropagation(probability_batch,discounted_rewards)