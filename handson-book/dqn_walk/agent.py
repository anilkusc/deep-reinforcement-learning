import torch
from torch import nn
from torch.nn import Sequential,Linear,ReLU
from tensorboardX import SummaryWriter
import numpy as np
import random
from collections import deque 

class Agent(nn.Module):
    def __init__(self,input=32,output=4,hidden=256,conv_hidden=32,learning_rate=0.001,gamma = 0.99,max_epsilon=1.0,min_epsilon = 0.01,exploration_decay_rate=0.000001,replay_memory=5000,memory_batch_size=1000,tensorboard=False):
        super(Agent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        self.input = input
        self.output = output
        self.learning_rate = learning_rate
        self.hidden=hidden
        self.conv_hidden = conv_hidden
        self.replay_memory = deque(maxlen=replay_memory)
        self.transitions = []
        self.tensorboard = tensorboard
        self.model_constructor()
        self.gamma = gamma
        self.epsilon = max_epsilon
        self.memory_batch_size = memory_batch_size
        self.episode = 1
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon        
        self.exploration_decay_rate = exploration_decay_rate
    def model_constructor(self):

        self.model = Sequential(
            Linear(self.input, self.hidden),
            ReLU(),
            Linear(self.hidden, self.hidden),
            ReLU(),
            Linear(self.hidden, self.output)
        ).to(self.device)

        self.target_model = Sequential(
            Linear(self.input, self.hidden),
            ReLU(),
            Linear(self.hidden, self.hidden),
            ReLU(),
            Linear(self.hidden, self.output)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.tensorboard:
            self.writer = SummaryWriter(comment="-dqn-pong")
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def compute_loss(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        current_q_values = self.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.forward_target_model(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        loss = torch.nn.functional.mse_loss(current_q_values, target_q_values)
        return loss

    def train(self,batches):
        self.optimizer.zero_grad()
        state_batch,action_batch,reward_batch,next_state_batch,done_batch = batches
        #actual_Q = self.forward(state_batch)
        #unsq_act_batch = action_batch.unsqueeze(-1)
        #actual_Q = actual_Q.gather(1,unsq_act_batch )
        #actual_Q = actual_Q.squeeze(-1)
        #with torch.no_grad():
        #    next_state_values = self.forward_target_model(next_state_batch).max(1)[0]
        #    next_state_values[done_batch] = 0.0
        #    next_state_values = next_state_values.detach()
        #expected_Q = reward_batch + self.gamma * next_state_values
        #loss = MSELoss()(actual_Q,expected_Q)
        loss = self.compute_loss(state_batch,action_batch,reward_batch,next_state_batch,done_batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def get_batches(self):
        if len(self.replay_memory) > self.memory_batch_size:
            minibatch = random.sample(self.replay_memory, self.memory_batch_size)

            # Convert numpy arrays to tensors before stacking
            state_batch = torch.stack([s1 for (r, s1, a, s2, d) in minibatch]).to(self.device)
            action_batch = torch.tensor([a for (r, s1, a, s2, d) in minibatch], dtype=torch.long).to(self.device)
            reward_batch = torch.tensor([r for (r, s1, a, s2, d) in minibatch], dtype=torch.float).to(self.device)
            next_state_batch = torch.stack([s2 for (r, s1, a, s2, d) in minibatch]).to(self.device)
            #done_batch = torch.tensor([d for (r, s1, a, s2, d) in minibatch], dtype=torch.bool).to(self.device)
            done_batch = torch.tensor([1 if d else 0 for (r, s1, a, s2, d) in minibatch], dtype=torch.float).to(self.device)

            return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        return None

    def action_selector(self, state):
        if random.random() < self.epsilon:
            action = np.random.randint(0, self.output)
        else:
            if isinstance(state, list):
                state = np.array(state)
            state_ = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            QValues_ = self.forward(state_)
            QValues = QValues_.cpu().data.numpy()[0]
            action = np.argmax(QValues)
        return action
    
    def update(self):
        return self.train(self.get_batches())
    
    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size()[0], -1)
        return self.model(conv_out)

    def forward_target_model(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size()[0], -1) 
        return self.target_model(conv_out)

    def epsilon_decay(self):
        self.epsilon = max(self.min_epsilon, self.max_epsilon * (0.99 ** self.episode))
        self.episode += 1

    def save(self):
        torch.save(self.model.state_dict(), "model_ac.pth")

    def load(self):
        self.model.load_state_dict(torch.load("model_ac.pth", map_location=self.device))