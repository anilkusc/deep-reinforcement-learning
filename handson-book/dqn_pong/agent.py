import torch
from torch import nn
from torch.nn import Sequential,Linear,ReLU,MSELoss,Softmax,Conv2d
from tensorboardX import SummaryWriter
import numpy as np
import random
from collections import deque 

class Agent(nn.Module):
    def __init__(self,input=32,output=4,learning_rate=0.001,gamma = 0.99,epsilon=1.0,replay_memory=10000,memory_batch_size=1000,tensorboard=False):
        super(Agent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input = input
        self.output = output
        self.learning_rate = learning_rate
        self.replay_memory = deque(maxlen=replay_memory)
        self.transitions = []
        self.tensorboard = tensorboard
        self.model_constructor()
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory_batch_size = memory_batch_size
        self.episode = 1

    def model_constructor(self):

        self.conv = Sequential(
            Conv2d(self.input[0], 32, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(32, 64, kernel_size=4, stride=2),
            ReLU(),
            Conv2d(64, 64, kernel_size=3, stride=1),
            ReLU()
        )
        conv_out_size = self._get_conv_out()
        self.model = Sequential(
            Linear(conv_out_size, 512),
            ReLU(),
            Linear(512, self.output)
        ).to(self.device)

        #self.target_model = Sequential(
        #    Linear(conv_out_size, 512),
        #    ReLU(),
        #    Linear(512, self.output)
        #).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.tensorboard:
            self.writer = SummaryWriter(comment="-dqn-pong")

    def train(self,batches):
        self.optimizer.zero_grad()
        state_batch,action_batch,reward_batch,next_state_batch,done_batch = batches
        actual_Q = self.model(state_batch).gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.model(next_state_batch).max(1)[0]
            next_state_values[done_batch] = 0.0
            next_state_values = next_state_values.detach()
        expected_Q = reward_batch + self.gamma * next_state_values

        loss = MSELoss()(actual_Q,expected_Q)
        loss.backward()
        self.optimizer.step()
        if self.tensorboard:
            self.writer.add_scalar("loss", loss.item(), self.episode)
    
    def monitor(self,reward,loss,i):
        self.writer.add_scalar("loss", loss.item(), i)
        self.writer.add_scalar("reward", reward, i)

    def action_selector(self,state):
        state_ = torch.FloatTensor([state])
        if (random.random() < self.epsilon):
            action= np.random.randint(0,self.output)
        else:
            QValues_ = self.forward(state_)
            QValues = QValues_.data.numpy()[0]
            action = np.argmax(QValues)
        return action

    def _get_conv_out(self):
        o = self.conv(torch.zeros(1, *self.input))
        return int(np.prod(o.size()))

    def update(self):
        self.epsilon_decay()
        batches = self.get_batches()
        if batches != None:
            self.train(batches)
        self.save()

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.model(conv_out)

    def epsilon_decay(self):
        min_epsilon = 0.01
        exploration_decay_rate = 0.01
        self.epsilon = min_epsilon + (self.epsilon - min_epsilon) * np.exp(-exploration_decay_rate*self.episode)
        self.episode += 1

    def get_batches(self):
        if len(self.replay_memory) > self.memory_batch_size:
            minibatch = random.sample(self.replay_memory, self.memory_batch_size)

            # Convert numpy arrays to tensors before stacking
            state_batch = torch.stack([torch.from_numpy(s1) for (r, s1, a, s2, d) in minibatch]).to(self.device)
            action_batch = torch.tensor([a for (r, s1, a, s2, d) in minibatch], dtype=torch.int).to(self.device)
            reward_batch = torch.tensor([r for (r, s1, a, s2, d) in minibatch], dtype=torch.float).to(self.device)
            next_state_batch = torch.stack([torch.from_numpy(s2) for (r, s1, a, s2, d) in minibatch]).to(self.device)
            done_batch = torch.tensor([d for (r, s1, a, s2, d) in minibatch], dtype=torch.bool).to(self.device)

            return state_batch, action_batch, reward_batch, next_state_batch, done_batch
        return None


    def save(self):
        torch.save(self.model.state_dict(), "model_ac.pth")

    def load(self):
        self.model.load_state_dict(torch.load("model_ac.pth", map_location=self.device))