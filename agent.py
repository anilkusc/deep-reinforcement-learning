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
        self.learning_rate = learning_rate
        self.sync_freq = sync_freq
        self.sync_counter = 0
        self.memory_batch_size = memory_batch_size
        self.replay_memory = deque(maxlen=replay_memory)
        self.transitions = []
        #self.model_constructor()
        self.actor_constructor()
        self.critic_constructor()

        self.gamma = gamma
        self.epsilon = epsilon

    def model_constructor(self):
        self.model = torch.nn.Sequential(
        torch.nn.Linear(self.input, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, self.output),
        torch.nn.Softmax()
            ).to(self.device)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())

    def actor_constructor(self):
        self.actor = torch.nn.Sequential(
        torch.nn.Linear(self.input, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, self.output),
        torch.nn.Softmax(dim=-1)
            ).to(self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

    def critic_constructor(self):
        self.critic = torch.nn.Sequential(
        torch.nn.Linear(self.input, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, 1)
            ).to(self.device)
        self.loss_fn_critic = torch.nn.MSELoss()
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)


    def train_backpropagation(self,X,Y):
        X, Y = X.to(self.device), Y.to(self.device)
        #Y is real reward value
        #X is predicted reward value
        self.optimizer.zero_grad()
        #loss = self.loss_fn(X,Y)
        #loss = self.loss_fn_preds(X,Y)
        loss = self.loss_fn_advantage(X,Y)
        loss.backward()
        self.optimizer.step()
        return loss


    def loss_fn_preds(self,predictions,rewards):
        # We are taking logaritmic because this will penalty more if predictions close to 0.
        # For example if model find probability as very close to 0.(0.01) But it should be more. We are penalty this attidude more with logaritmic function.
        total_rewards = torch.sum(rewards*torch.log(predictions)).to(self.device)
        #return total_rewards
        # Need to study more about why we multiply it with -1 :D 
        return -1 * total_rewards

    def state_to_tensor_input(self,state):
        #normalized_state = state / float(input_size-1)
        nn_input = np.zeros(self.input)
        nn_input[state] = 1.0
        pytorch_input = torch.from_numpy(nn_input).float().to(self.device)
        return pytorch_input

    def state_to_flatten(self,state):
        return torch.from_numpy(state).float().to(self.device)

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
    
    def action_dist(self,state_):
        probs = self.actor(state_)
        dist = torch.distributions.Categorical(probs=probs)
        return dist 
    
    def discount_rewards(self,rewards):
        # it is going to create [N,N-1,... 3,2,1] array for multiplying it with gamma 
        discount_batch = torch.pow(self.gamma,torch.arange(len(rewards)).float()).to(self.device)
        # If first move most important you do not need to flip. But reward is at the and it is better to flip it because the most important move is at the last.
        discount_batch_flipped = torch.flip(discount_batch, dims=(0,))
        discount_return =  discount_batch_flipped * rewards
        #normalize reward
        #discount_return /= discount_return.max()
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

    def calculate_advantage(self,reward,state,next_state,done):
        value = self.critic(state)
        next_value = self.critic(next_state)
        advantage = reward + ((1-done) * self.gamma * next_value) - value
        return advantage
    
    def save(self):
        torch.save(self.actor.state_dict(), "model_ac.pth")

    def load(self):
        self.actor.load_state_dict(torch.load("model_ac.pth", map_location=self.device))
    
    def batch_state_transition(self):
        minibatch = self.transitions    
        reward_batch = torch.Tensor([r for (s,a,r) in minibatch]).to(self.device)
        state_batch = torch.stack([s for (s,a,r) in minibatch]).to(self.device)
        action_batch = torch.Tensor([a for (s,a,r) in minibatch]).to(self.device)
        # gerçekleşen her eylemin reward ı bulunur
        discounted_rewards = self.discount_rewards(reward_batch)
        # gerçekleşen her eylemin modele göre olma olasılığı bulunur
        prediction_batch = self.model(state_batch)
        #prediction_batch = [[0.1, 0.7, 0.2],  # İlk durum için olasılıklar
        #            [0.3, 0.3, 0.4]]  # İkinci durum için olasılıklar
        #action_batch = [1, 2]  # Gerçekleşen eylemler (0-tabanlı indeks)
        #probability_batch = [0.7, 0.4]  # Seçilen eylemlerin olasılıkları
        probability_batch = prediction_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()
        return probability_batch,discounted_rewards

    def batch_state_replay_memory(self,minibatch):
        state_batch = torch.stack([s1 for (r, s1, a, s2, d) in minibatch]).to(self.device)
        action_batch = torch.Tensor([a for (r, s1, a, s2, d) in minibatch]).to(self.device)
        reward_batch = torch.Tensor([r for (r, s1, a, s2, d) in minibatch]).to(self.device)
        next_state_batch = torch.stack([s2 for (r, s1, a, s2, d) in minibatch]).to(self.device)
        done_batch = torch.Tensor([float(d) for (r, s1, a, s2, d) in minibatch]).to(self.device)
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

    def train_backpropagation_actor(self,dist,action,advantage):
        
        loss_actor = -dist.log_prob(action)*advantage.detach()
        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()
        return loss_actor

    def train_backpropagation_critic(self, advantage):
        loss_critic = advantage.pow(2).mean()
        self.optimizer_critic.zero_grad()  # Zero the gradients
        loss_critic.backward()  # Compute gradients
        self.optimizer_critic.step()  # Update the critic network

        return loss_critic
    
    def loss_fn_advantage(self, policy_logits, action, advantage):
        log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        
        action = action.long() 
        action_log_probs = log_probs.gather(1, action.unsqueeze(1)).squeeze(1) 
        
        actor_loss = -action_log_probs * advantage
        
        return actor_loss.sum()
    
    def update_actor_critic(self,reward,state,dist,action,next_state,done):
        advantage = self.calculate_advantage(reward,state,next_state,done)
        actor_loss = self.train_backpropagation_actor(dist,action,advantage)
        critic_loss = self.train_backpropagation_critic(advantage)
        return actor_loss, critic_loss

    def update_actor_critic_trajectory(self,trajectories):
        state_batch,action_batch,reward_batch,next_state_batch,done_batch = self.batch_state_replay_memory(trajectories)
        advantage = self.calculate_advantage(reward_batch,state_batch,next_state_batch,done_batch)
        actor_loss = self.train_backpropagation_actor(state_batch,action_batch,advantage)
        critic_loss = self.train_backpropagation_critic(state_batch,advantage)
        return actor_loss, critic_loss

    def REINFORCE(self):
        probability_batch,discounted_rewards = self.batch_state_transition()
        return self.train_backpropagation(probability_batch,discounted_rewards)