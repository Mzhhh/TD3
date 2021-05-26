import numpy as np
import torch
import os
import re


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, *state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, *state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
	
	def get_batch(self, batch_index):

		return (
			torch.FloatTensor(self.state[batch_index]).to(self.device),
			torch.FloatTensor(self.action[batch_index]).to(self.device),
			torch.FloatTensor(self.next_state[batch_index]).to(self.device),
			torch.FloatTensor(self.reward[batch_index]).to(self.device),
			torch.FloatTensor(self.not_done[batch_index]).to(self.device)
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
	  
	
		
		
