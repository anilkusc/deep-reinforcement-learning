import torch
from torch import nn
from torch.nn import Sequential,Linear,ReLU,CrossEntropyLoss,Softmax,Conv2d
from tensorboardX import SummaryWriter
import numpy as np
import random
class Agent(nn.Module):
    def __init__(self,input=32,output=4,learning_rate=0.001,gamma = 0.99,epsilon=1.0,replay_memory=10000,tensorboard=False):
        super(Agent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input = input
        self.output = output
        self.learning_rate = learning_rate
        self.replay_memory = []
        self.transitions = []
        self.tensorboard = tensorboard
        self.model_constructor()
        self.gamma = gamma
        self.epsilon = epsilon

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

        self.target_model = Sequential(
            Linear(conv_out_size, 512),
            ReLU(),
            Linear(512, self.output)
        ).to(self.device)

        self.lossfunc = CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.tensorboard:
            self.writer = SummaryWriter(comment="-dqn-pong")
        self.softmax = Softmax(dim=1)

    def train(self,x,y):
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.lossfunc(out,y)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def monitor(self,reward,loss,i):
        self.writer.add_scalar("loss", loss.item(), i)
        self.writer.add_scalar("reward", reward, i)

    def action_selector(self,state):
        state_ = torch.FloatTensor([state])
        #if (random.random() < self.epsilon):
        #    action= np.random.randint(0,self.output)
        #else:
        QValues_ = self.forward(state_)
        QValues = QValues_.data.numpy()[0]
        action = np.argmax(QValues)
        return action

    def _get_conv_out(self):
        o = self.conv(torch.zeros(1, *self.input))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.model(conv_out)

    def save(self):
        torch.save(self.model.state_dict(), "model_ac.pth")

    def load(self):
        self.model.load_state_dict(torch.load("model_ac.pth", map_location=self.device))