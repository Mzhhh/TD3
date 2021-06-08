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


latent_dim = 1024
z_dim = 32

log_freq = 100


class Actor(nn.Module):

	def __init__(self, img_channels, action_dim, min_action, max_action):

		super(Actor, self).__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(img_channels, 32, kernel_size=4, stride=2),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2), 
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2), 
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=4, stride=2), 
			nn.BatchNorm2d(256),
			nn.ReLU(),
			Flatten()
		)

		self.linear = nn.Sequential(
			nn.Linear(latent_dim, z_dim),
			nn.ReLU(),
			nn.Linear(z_dim, action_dim)
		)
		
		assert min_action.size == action_dim and max_action.size == action_dim
		assert np.all(max_action > min_action)
		self.mid_action = torch.from_numpy(((min_action+max_action)/2).copy()).to(device)
		self.range_action = torch.from_numpy(((max_action-min_action)/2).copy()).to(device)
	
	
	def forward(self, state):
		a = self.encoder(state)
		a = self.linear(a)
		return self.mid_action + self.range_action * torch.tanh(a)


class Critic(nn.Module):
	def __init__(self, img_channels, action_dim):
		super(Critic, self).__init__()

		self.encoder = nn.Sequential(
			nn.Conv2d(img_channels, 32, kernel_size=4, stride=2),  
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2), 
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2), 
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=4, stride=2),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			Flatten() 
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
		state = torch.FloatTensor(state).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=64):
		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		state = state.view(-1, 3, 64, 64)
		next_state = next_state.view(-1, 3, 64, 64)
		
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
		
	
	def critic_loss(self, batch_data, writer=None, t=0):

		# Sample replay buffer 
		state, action, next_state, reward, not_done = batch_data
		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()
		# Get current Q estimate
		current_Q = self.critic(state, action)
		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		if writer and (t+1) % log_freq == 0:
			with torch.no_grad():
				writer.add_scalar('expertQ_train_raw/critic_loss', critic_loss, t + 1)
				writer.add_scalar('expertQ_train_raw/not_done', not_done.sum(), t + 1)

				# current_Q
				writer.add_scalar('expertQ_train_raw/current_Q/mean', torch.mean(current_Q), t + 1)
				writer.add_scalar('expertQ_train_raw/current_Q/max', current_Q.max(), t + 1)
				writer.add_scalar('expertQ_train_raw/current_Q/min', current_Q.min(), t + 1)
				writer.add_scalar('expertQ_train_raw/current_Q/std', torch.std(current_Q), t + 1)

				# # next_Q
				# # next_Q = torch.masked_select(next_Q, torch.BoolTensor(states[:, -self.a_dim:]))
				# writer.add_scalar('expertQ_train_raw/next_Q/mean', torch.mean(next_Q), t + 1)
				# writer.add_scalar('expertQ_train_raw/next_Q/max', next_Q.max(), t + 1)
				# writer.add_scalar('expertQ_train_raw/next_Q/min', next_Q.min(), t + 1)
				# writer.add_scalar('expertQ_train_raw/next_Q/std', torch.std(next_Q), t + 1)
				#
				# # target_next_Q
				# # target_next_Q = torch.masked_select(target_next_Q, torch.BoolTensor(next_states[:, -self.a_dim:]))
				# writer.add_scalar('expertQ_train_raw/future_target_Q/mean', torch.mean(target_next_Q), t + 1)
				# writer.add_scalar('expertQ_train_raw/future_target_Q/max', target_next_Q.max(), t + 1)
				# writer.add_scalar('expertQ_train_raw/future_target_Q/min', target_next_Q.min(), t + 1)
				# writer.add_scalar('expertQ_train_raw/future_target_Q/std', torch.std(target_next_Q), t + 1)
				#
				# # next_Q_diff
				# next_Q_diff = torch.max(next_Q, 1)[0] - target_next_Q
				# writer.add_scalar('expertQ_train_raw/next_Q_diff/mean', torch.mean(next_Q_diff), t + 1)
				# writer.add_scalar('expertQ_train_raw/next_Q_diff/max', next_Q_diff.max(), t + 1)
				# writer.add_scalar('expertQ_train_raw/next_Q_diff/min', next_Q_diff.min(), t + 1)
				# writer.add_scalar('expertQ_train_raw/next_Q_diff/std', torch.std(next_Q_diff), t + 1)

				# expected_Q
				writer.add_scalar('expertQ_train_raw/target_Q/mean', torch.mean(target_Q), t + 1)
				writer.add_scalar('expertQ_train_raw/target_Q/max', target_Q.max(), t + 1)
				writer.add_scalar('expertQ_train_raw/target_Q/min', target_Q.min(), t + 1)
				writer.add_scalar('expertQ_train_raw/target_Q/std', torch.std(target_Q), t + 1)
		
		return critic_loss


	def actor_loss(self, batch_data):

		# Sample replay buffer 
		state, action, next_state, reward, not_done = batch_data
		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()

		return actor_loss


	def Q_value(self, batch_data):
		
		# Sample replay buffer 
		state, action, next_state, reward, not_done = batch_data
		# Get current Q estimate
		current_Q = self.critic(state, action)

		return current_Q
  
	
	def clear_gradient(self):
		self.actor_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
  
  
	def update_weights(self):
	
		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
  