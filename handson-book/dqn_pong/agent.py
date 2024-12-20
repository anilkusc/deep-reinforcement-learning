import torch
from torch import nn
from torch.nn import Sequential,Linear,LeakyReLU,ReLU,MSELoss,Conv2d,MaxPool2d
from tensorboardX import SummaryWriter
import numpy as np
import random
from collections import deque 

class Agent(nn.Module):
    def __init__(self,input=32,output=4,hidden=256,conv_hidden=32,learning_rate=0.001,gamma = 0.99,max_epsilon=1.0,min_epsilon = 0.01,exploration_decay_rate=0.990,replay_memory=5000,memory_batch_size=1000,tensorboard=False):
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

        self.conv = Sequential(
            Conv2d(self.input[0], self.conv_hidden, kernel_size=8, stride=4),
            ReLU(),
            #MaxPool2d(kernel_size=2, stride=2),  

            Conv2d(self.conv_hidden, self.conv_hidden * 2, kernel_size=4, stride=2),
            ReLU(),
            #MaxPool2d(kernel_size=2, stride=2),

            Conv2d(self.conv_hidden * 2, self.conv_hidden * 2, kernel_size=3, stride=1, padding=1),
            ReLU()
        ).to(self.device)

        conv_out_size = self._get_conv_out()

        self.model = Sequential(
            Linear(conv_out_size, self.hidden),
            LeakyReLU(),
            Linear(self.hidden, self.output)
        ).to(self.device)
        
        self.model.apply(lambda m: torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu') if isinstance(m, nn.Linear) else None)

        self.target_model = Sequential(
            Linear(conv_out_size, self.hidden),
            LeakyReLU(),
            Linear(self.hidden, self.output)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.tensorboard:
            self.writer = SummaryWriter(comment="-dqn-pong")
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def compute_loss(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):

        current_q_value = self.forward(state_batch)[0][action_batch]

        with torch.no_grad():
            # Target Q-values
            next_q_values = self.forward_target_model(next_state_batch)[0]
            next_q_value = torch.max(next_q_values)
            next_q_value = next_q_value.detach()
            target_q_value = reward_batch + (1 - done_batch) * self.gamma * next_q_value

        # Loss computation
        loss = torch.nn.functional.mse_loss(current_q_value, target_q_value)
        return loss
    

    def compute_loss2(self):
        states, actions, rewards, dones, next_states = self.sample()
        states_v = torch.tensor(np.array(states, copy=False)).to(self.device)
        next_states_v = torch.tensor(np.array(next_states, copy=False)).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)

        state_action_values = self.forward(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.forward_target_model(next_states_v).max(1)[0]
            next_state_values[done_mask] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards_v
        return MSELoss()(state_action_values,expected_state_action_values)

    def train(self):
        self.optimizer.zero_grad()
        #state_batch,action_batch,reward_batch,next_state_batch,done_batch = batches
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
        #loss = self.compute_loss(state_batch,action_batch,reward_batch,next_state_batch,done_batch)
        loss = self.compute_loss2()
        loss.backward()
        self.optimizer.step()
        return loss

    def get_batches(self):
        if len(self.replay_memory) > self.memory_batch_size:
            minibatch = random.sample(self.replay_memory, self.memory_batch_size)

            # Convert numpy arrays to tensors before stacking
            state_batch = torch.stack([torch.from_numpy(s1).float() for (r, s1, a, s2, d) in minibatch]).to(self.device)
            action_batch = torch.tensor([a for (r, s1, a, s2, d) in minibatch], dtype=torch.long).to(self.device)
            reward_batch = torch.tensor([r for (r, s1, a, s2, d) in minibatch], dtype=torch.float).to(self.device)
            next_state_batch = torch.stack([torch.from_numpy(s2).float() for (r, s1, a, s2, d) in minibatch]).to(self.device)
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
            #print("0:"+str(QValues[0])+"1:"+str(QValues[1])+"2:"+str(QValues[2])+"3:"+str(QValues[3])+"4:"+str(QValues[4])+"5:"+str(QValues[5]))
            self.writer.add_scalar("qval0", QValues[0], self.episode)
            self.writer.add_scalar("qval1", QValues[1], self.episode)
            #self.writer.add_scalar("qval2", QValues[2], self.episode)
            #self.writer.add_scalar("qval3", QValues[3], self.episode)
            #self.writer.add_scalar("qval4", QValues[4], self.episode)
            #self.writer.add_scalar("qval5", QValues[5], self.episode)
        return action
    
    def update(self):
        #return self.train(self.get_batches())
        return self.train()
    def update_step(self,state, action, reward, next_state, done):
        self.optimizer.zero_grad()
    
        loss = self.compute_loss(
            torch.from_numpy(state).float().to(self.device),       # state için float32 dönüşümü
            torch.tensor(action, dtype=torch.long).to(self.device),  # action için long dönüşümü
            torch.tensor(reward, dtype=torch.float32).to(self.device),  # reward için float32 dönüşümü
            torch.from_numpy(next_state).float().to(self.device),  # next_state için float32 dönüşümü
            torch.tensor(1 if done else 0, dtype=torch.float32).to(self.device)  # done için float32 dönüşümü
        )        
        loss.backward()
        self.optimizer.step()
        return loss

    def _get_conv_out(self):
        o = self.conv(torch.zeros(1, *self.input,device=self.device))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.model(conv_out)

    def forward_target_model(self, x):
        x = x.to(dtype=torch.float32)
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size()[0], -1) 
        return self.target_model(conv_out)

    def epsilon_decay(self):
        self.epsilon = max(self.min_epsilon, self.max_epsilon * (self.exploration_decay_rate ** self.episode))
        self.episode += 1

    def save(self):
        torch.save(self.model.state_dict(), "model_ac.pth")

    def load(self):
        self.model.load_state_dict(torch.load("model_ac.pth", map_location=self.device))

    def sample(self):
        indices = np.random.choice(len(self.replay_memory), self.memory_batch_size,replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.replay_memory[idx] for idx in indices])
        return np.array(states), np.array(actions),np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)