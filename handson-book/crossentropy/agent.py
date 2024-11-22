import torch
from torch.nn import Sequential,Linear,ReLU,CrossEntropyLoss,Softmax
from tensorboardX import SummaryWriter
import numpy as np
class Agent():
    def __init__(self,input=32,output=4,learning_rate=0.001,gamma = 0.99,epsilon=1.0,replay_memory=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input = input
        self.output = output
        self.learning_rate = learning_rate
        self.replay_memory = []
        self.transitions = []
        self.model_constructor()
        self.gamma = gamma
        self.epsilon = epsilon

    def model_constructor(self):
        self.model = Sequential(
        Linear(self.input, 128),
        ReLU(),
        Linear(128, self.output)
            ).to(self.device)
        self.lossfunc = CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.writer = SummaryWriter(comment="-cartpole")
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

    def save(self):
        torch.save(self.model.state_dict(), "model_ac.pth")

    def load(self):
        self.model.load_state_dict(torch.load("model_ac.pth", map_location=self.device))

    def action_selector(self,state):
        state_ = torch.FloatTensor([state])
        probs = self.softmax(self.model(state_))
        #convert tensor to numpy
        dist = probs.data.numpy()[0]
        action = np.random.choice(len(dist), p=dist)
        return action
    
    def filter_best_trajectories_as_tensor(self,trajectories,percentage=0.3):
        sorted_data = sorted(trajectories, key=lambda x: x["return"], reverse=True)
        cut_point = int(len(sorted_data) * percentage)
        best_trajectories = sorted_data[:cut_point]
        train_trajectories = []
        train_actions = []
        for t in best_trajectories:
            for episode in t["episodes"]:
                train_trajectories.append(episode[0])
                train_actions.append(episode[1])
        train_obs_v = torch.FloatTensor(train_trajectories)
        train_act_v = torch.LongTensor(train_actions)
        return train_obs_v,train_act_v