import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


latent_dim = 4096
z_dim = 32


class Actor(nn.Module):

    def __init__(self, img_channels, action_dim, min_action, max_action):

        super(Actor, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4,
                      stride=2),  # b, 32, 47, 47
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # b, 64, 22, 22
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),  # b, 128, 10, 10
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),  # b, 256, 4, 4
            nn.ReLU(),
            Flatten()  # b, 4096
        )

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, action_dim)
        )
        
        assert min_action.size == action_dim and max_action.size == action_dim
        assert np.all(max_action > min_action)
        self.mid_action = (min_action + max_action) / 2
        self.range_action = (max_action - min_action) / 2
    
    def forward(self, state):
        a = self.encoder(state)
        a = self.linear(a)
        return self.mid_action + self.range_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, img_channels, action_dim):
        super(Critic, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=4,
                      stride=2),  # b, 32, 47, 47
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # b, 64, 22, 22
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),  # b, 128, 10, 10
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),  # b, 256, 4, 4
            nn.ReLU(),
            Flatten()  # b, 4096
        )
        
        self.linear = nn.Sequential(
            nn.Linear(latent_dim+action_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, 1)
        )

    def forward(self, state, action):
        q = self.encoder(state)
        q = self.linear(torch.cat([q, action], 1))
        return q
    

class DDPG(object):
    def __init__(self, img_channels, action_dim, min_action, max_action, discount=0.99, tau=0.001):
        self.actor = Actor(img_channels, action_dim, min_action, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(img_channels, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.discount = discount
        self.tau = tau


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=64):
        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Get current Q estimate
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        # Optimize the actor 
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        
