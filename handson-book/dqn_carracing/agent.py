import torch
from torch import nn
from torch.nn import Sequential,Linear,ReLU,MSELoss,Conv2d,MaxPool2d
from tensorboardX import SummaryWriter
import numpy as np
import random
from collections import deque 

class Agent(nn.Module):
    def __init__(self,input=32,output=4,hidden=256,conv_hidden=32,learning_rate=0.0001,gamma = 0.99,max_epsilon=1.0,min_epsilon = 0.01,exploration_decay_rate=0.990,replay_memory=5000,memory_batch_size=1000,tensorboard=False):
        super(Agent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.conv = Sequential(
            Conv2d(self.input[0], self.conv_hidden, kernel_size=8, stride=4),
            ReLU(),
            Conv2d(self.conv_hidden, self.conv_hidden * 2, kernel_size=4, stride=2),
            ReLU()
            ).to(self.device)
        conv_out_size = self._get_conv_out()

        self.model = nn.Sequential(
            Linear(conv_out_size, self.hidden),
            ReLU(),
            Linear(self.hidden, self.output)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.tensorboard:
            self.writer = SummaryWriter(comment="-dqn-carracing")

    def compute_loss(self):
        state_batch ,action_batch ,reward_batch ,next_state_batch ,done_batch = self.get_batches()
        current_q_value = self.model_forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.model_forward(next_state_batch).max(1)[0]
            target_q_value = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        return MSELoss()(current_q_value, target_q_value)
    
    def train(self):
        self.optimizer.zero_grad()
        loss = self.compute_loss()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def action_selector(self, state):
        if random.random() < self.epsilon:
            action = np.random.randint(0, self.output)
        else:
            if isinstance(state, list):
                state = np.array(state)
            #state_ = torch.tensor(state, dtype=torch.float32).to(self.device)
            state_ = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            QValues_ = self.model_forward(state_)
            QValues = QValues_.cpu().data.numpy()[0]
            action = np.argmax(QValues)
            if self.tensorboard:
                for i in range(len(QValues)):
                    self.writer.add_scalar(f"QValues/action_{i}", QValues[i], self.episode)
            #for i in range(len(QValues)):
            #    print("Q Value " + str(i) + " : " + str(QValues[i]))
        return action

    def epsilon_decay(self):
        self.epsilon = max(self.min_epsilon, self.max_epsilon * (self.exploration_decay_rate ** self.episode))
        self.episode += 1

    def save(self, model_name="model_ac.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'conv_model_state_dict': self.conv.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_architecture': self.model,
        }, model_name)

    def load(self, model_name="model_ac.pth"):
        checkpoint = torch.load(model_name, map_location=self.device)
        if 'model_architecture' in checkpoint:
            self.model = checkpoint['model_architecture']
        else:
            exit()

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.conv.load_state_dict(checkpoint['conv_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def get_batches(self):
        minibatch = random.sample(self.replay_memory, self.memory_batch_size)
        state_batch = torch.stack([torch.tensor(s1, dtype=torch.float) for (r, s1, a, s2, d) in minibatch]).to(self.device)
        action_batch = torch.tensor([a for (r, s1, a, s2, d) in minibatch], dtype=torch.long).to(self.device)
        reward_batch = torch.tensor([r for (r, s1, a, s2, d) in minibatch], dtype=torch.float).to(self.device)
        next_state_batch = torch.stack([torch.tensor(s2, dtype=torch.float) for (r, s1, a, s2, d) in minibatch]).to(self.device)
        done_batch = torch.tensor([1 if d else 0 for (r, s1, a, s2, d) in minibatch], dtype=torch.float).to(self.device)

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def _get_conv_out(self):
        o = self.conv(torch.zeros(1, *self.input,device=self.device))
        return int(np.prod(o.size()))

    def model_forward(self, x):
        # Flatten the convolutional output
        conv_out = self.conv(x)
        conv_out_flattened = conv_out.view(conv_out.size(0), -1)
        return self.model(conv_out_flattened)
    
    def target_model_forward(self, x):
        # Flatten the convolutional output
        conv_out = self.conv(x)
        conv_out_flattened = conv_out.view(conv_out.size(0), -1)
        return self.target_model(conv_out_flattened)

def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)