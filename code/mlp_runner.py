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
			transforms.Resize((224,224)),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])


all_transforms = transforms.Compose([
			transforms.Resize((224,224)),
			# transforms.RandomRotation(degrees=15),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
			transforms.RandomAffine(degrees=20, translate=(0.3,0.3), scale=(0.5,1.5), shear=None, resample=False, fillcolor=0),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

train_dataset = datasets.ImageFolder(
		traindir,
		all_transforms)

val_dataset = datasets.ImageFolder(
		valdir,
		val_transforms)
		# val_transforms)


model = models.resnext101_32x8d(pretrained=True)
# model.fc = nn.Linear(2048,512)

# for param in model.parameters():
# 	param.requires_grad = False

mlp = MLP(model.fc.out_features)
# mlp = MLP(1000)

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
	batch_size=40, 
	shuffle=True,
	num_workers=2)

val_loader = torch.utils.data.DataLoader(
	val_dataset, 
	batch_size=32,
	shuffle=False,
	num_workers=2)



l_rate=0.00001
optimizer = optim.Adam(
	list(model.parameters()) + 
	list(mlp.parameters()), lr=l_rate,weight_decay=0.00001)

# optimizer = optim.SGD(list(model.parameters()) + 
# 	list(mlp.parameters()), lr=0.1, momentum=0.9)

criterion = nn.CrossEntropyLoss()

ep = 100

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# model.eval()

for x in range(1,ep):

	losses = AverageMeter('Loss', ':.6f')
	# scheduler.step()
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
		if k % 20 == 0:
			print("\tStep: ",i,losses.__str__())

	print("\nEpoch: ",x,losses.__str__())
	validate(val_loader,model,mlp,criterion,device)
	# validate(train_loader,model,mlp,criterion,device)
	torch.save(model.state_dict(), "./models/rx101-31")
	torch.save(mlp.state_dict(), "./models/mlp-31")



