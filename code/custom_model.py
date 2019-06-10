import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np 

class MLP(nn.Module):

	def __init__(self,inputsize):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(inputsize,1024)
		self.pr1 = nn.PReLU()
		self.bn1 = nn.BatchNorm1d(1024)

		self.fc2 = nn.Linear(1024,512)
		self.pr2 = nn.PReLU()
		self.bn2 = nn.BatchNorm1d(512)

		self.fc3 = nn.Linear(512,256)
		self.pr3 = nn.PReLU()
		self.bn3 = nn.BatchNorm1d(256)

		self.fc4 = nn.Linear(256,128)
		self.bn4 = nn.BatchNorm1d(128)

		self.fc7 = nn.Linear(128,5)
		
		
	def forward(self, x):
		x = self.pr1(self.fc1(x),)
		x = self.bn1(x)

		x = self.pr2(self.fc2(x))
		x = self.bn2(x)

		x = self.pr3(self.fc3(x))
		x = self.bn3(x)

		x = (self.fc4(x))
		# x = self.bn4(x)

		x = (self.fc7(x))
		
		return x

class MLP2(nn.Module):

	def __init__(self,inputsize):
		super(MLP2, self).__init__()
		self.fc1 = nn.Linear(inputsize,1024)
		self.bn1 = nn.BatchNorm1d(1024)

		self.fc2 = nn.Linear(1024,512)
		self.bn2 = nn.BatchNorm1d(512)

		self.fc3 = nn.Linear(512,256)
		self.bn3 = nn.BatchNorm1d(256)

		self.fc4 = nn.Linear(256,128)
		self.bn4 = nn.BatchNorm1d(128)

		self.fc5 = nn.Linear(128,64)
		self.bn5 = nn.BatchNorm1d(64)

		self.fc6 = nn.Linear(64,32)
		self.bn6 = nn.BatchNorm1d(32)

		self.fc7 = nn.Linear(32,5)
		
		
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = self.bn1(x)

		x = F.relu(self.fc2(x))
		x = self.bn2(x)

		x = F.relu(self.fc3(x))
		x = self.bn3(x)

		x = F.relu(self.fc4(x))
		x = self.bn4(x)

		x = F.relu(self.fc5(x))
		# x = self.bn5(x)

		x = F.relu(self.fc6(x))
		# x = self.bn6(x)

		x = (self.fc7(x))
		
		return x



