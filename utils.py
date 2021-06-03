import numpy as np
import torch
import os
import re


class ReplayBuffer(object):

	def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.device = device if device is not None else "cpu"

		self.state = torch.zeros((max_size, *state_dim), device=self.device)
		self.action = torch.zeros((max_size, action_dim), device=self.device)
		self.next_state = torch.zeros((max_size, *state_dim), device=self.device)
		self.reward = torch.zeros((max_size, 1), device=self.device)
		self.not_done = torch.zeros((max_size, 1), device=self.device)
		
		
	def add(self, state, action, next_state, reward, done):
		# permute: (96, 96, 3) -> (3, 96, 96)
		self.state[self.ptr] = torch.from_numpy(state).float().permute(2, 0, 1).to(self.device)
		self.action[self.ptr] = torch.from_numpy(np.array(action)).float().to(self.device)
		self.next_state[self.ptr] = torch.from_numpy(next_state).float().permute(2, 0, 1).to(self.device)
		self.reward[self.ptr] = torch.tensor(reward, device=self.device)
		self.not_done[self.ptr] = torch.tensor(1. - done, device=self.device)

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind]
		)
	
	def get_batch(self, batch_index):

		return (
			self.state[batch_index],
			self.action[batch_index],
			self.next_state[batch_index],
			self.reward[batch_index],
			self.not_done[batch_index]
		)


class NpyDataLoader(object):
	
	def __init__(self, file_folder, start_idx=None, end_idx=None, batch_size=16):
		
		all_files = os.listdir(file_folder)
		file_numbers = []
		for f in all_files:
			s = re.search("\d+.npy", f)
			file_numbers.append(f[s.start():s.end()-4])
		file_numbers = sorted(file_numbers)
		if start_idx is not None:
			file_numbers = [f for f in file_numbers if int(f) >= start_idx]
		if end_idx is not None:
			file_numbers = [f for f in file_numbers if int(f) < end_idx]
		
		self.file_list = [f"X_{f}.npy" for f in file_numbers]
		assert len(self.file_list) > 0, "No available data file"
		self.batch_size = batch_size
		self.file_path = file_folder
		
	def __iter__(self):
		file_idx, array_idx = 0, 0
		X = np.load(os.path.join(self.file_path, self.file_list[file_idx]))
		assert self.batch_size < X.shape[0], "Author is too lazy to support the case where batch_size > #data in array"
		while True:
			# enough data to yield from current array
			while array_idx + self.batch_size < X.shape[0]:
				ret = X[array_idx:array_idx+self.batch_size, :]
				array_idx += self.batch_size
				yield torch.from_numpy(ret)
			# not enough data to yield
			ret = X[array_idx:, :]
			file_idx += 1
			if file_idx < len(self.file_list):
				X = np.load(os.path.join(self.file_path, self.file_list[file_idx]))
				array_idx = self.batch_size - ret.shape[0]
				ret = np.concatenate([ret, X[:array_idx, :]], axis=0)
				yield torch.from_numpy(ret)
			else:
				return torch.from_numpy(ret)
	  
	
		
		
