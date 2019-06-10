import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models,datasets
import torchvision.transforms as transforms

from custom_model import MLP

from sklearn.metrics import f1_score,accuracy_score
from utils import AverageMeter
import numpy as np
import pandas as pd
import skimage.io as io
import os
import sys

import pretrainedmodels


'''
python runner.py [load/noload] [modelpath/junk]

'''

def validate(val_loader,model, mlp,criterion,device):

	model.eval()
	mlp.eval()
	pred = None
	gt = None

	for i, (inputs, label) in enumerate(val_loader):

		label = label.detach().cpu().numpy()

		output = model(inputs.to(device))
		output = mlp(output)
		output = F.softmax(output, dim = 1)

		output = torch.argmax(output,dim=1)
		output = output.detach().cpu().numpy()

		if pred is None:
			pred = output
		else:
			pred = np.concatenate((pred,output))

		if gt is None:
			gt = label
		else:
			gt = np.concatenate((gt,label))

	f1 = f1_score(gt,pred,average='weighted')
	acc = accuracy_score(gt,pred)

	print("F1:",f1,"ACC:",acc)



traindir = os.path.join("./", 't_classes')
valdir = os.path.join("./", 'v_classes')
print(traindir)


val_transforms = transforms.Compose([
			transforms.Resize((331,331)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])


all_transforms = transforms.Compose([
			transforms.Resize((331,331)),
			# transforms.RandomRotation(degrees=15),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
			transforms.RandomAffine(degrees=15, translate=(0.3,0.3), scale=(0.5,1.5), shear=None, resample=False, fillcolor=0),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])	
		])

train_dataset = datasets.ImageFolder(
		traindir,
		all_transforms)

val_dataset = datasets.ImageFolder(
		valdir,
		val_transforms)
		# val_transforms)


model_name = 'pnasnet5large' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

# model.fc = nn.Linear(2048,512)

# for param in model.parameters():
# 	print(param)
# 	param.requires_grad = False

# for name, param in model.named_parameters():
# 	if name == 'last_linear.weight' or name == 'last_linear.bias':
# 		param.requires_grad = True
# 	else:
# 		 param.requires_grad = False       

# exit()
# model.fc.requires_grad = True

mlp = MLP(1000)

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model = nn.DataParallel(model)

mlp.to(device)
mlp = nn.DataParallel(mlp)

if sys.argv[1] == "load":
	print("loading from ",sys.argv[2],sys.argv[3])
	model.load_state_dict(torch.load(sys.argv[2]))
	mlp.load_state_dict(torch.load(sys.argv[3]))

train_loader = torch.utils.data.DataLoader(
	train_dataset, 
	batch_size=8, 
	shuffle=True,
	num_workers=2)

val_loader = torch.utils.data.DataLoader(
	val_dataset, 
	batch_size=8,
	shuffle=False,
	num_workers=2)



l_rate=0.00001
optimizer = optim.Adam(
	list(model.parameters())+
	list(mlp.parameters()),
	 lr=l_rate,weight_decay=0.00001)

# optimizer = optim.SGD(list(model.parameters()) + 
# 	list(mlp.parameters()), lr=0.1, momentum=0.9)

criterion = nn.CrossEntropyLoss()

ep = 100

# model.eval()

for x in range(ep):

	validate(val_loader,model,mlp,criterion,device)

	losses = AverageMeter('Loss', ':.6f')
	k = -1
	for i, (inputs, label) in enumerate(train_loader):

		inputs, label = inputs.to(device), label.to(device)

		output = model(inputs)
		output = mlp(output)
		output = F.softmax(output, dim = 1)
		
		loss = criterion(output, label)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.update(loss.item(), inputs.size(0))

		k += 1
		if k % 10 == 0:
			print("\tStep: ",i,losses.__str__())

	print("\nEpoch: ",x,losses.__str__())
	
	# validate(train_loader,model,mlp,criterion,device)
	torch.save(model.state_dict(), "./models/pnasnet5large-29")
	torch.save(mlp.state_dict(), "./models/mlp-29")



