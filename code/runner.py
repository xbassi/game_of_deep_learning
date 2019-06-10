import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models,datasets
import torchvision.transforms as transforms

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

def validate(val_loader,model, criterion,device):

	model.eval()
	pred = None
	gt = None

	for i, (inputs, label) in enumerate(val_loader):

		label = label.detach().cpu().numpy()

		output = model(inputs.to(device))
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
			transforms.ToTensor()
		])


all_transforms = transforms.Compose([
			transforms.Resize((224,224)),
			# transforms.RandomRotation(degrees=10),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
			transforms.RandomAffine(degrees=20, translate=(0.4,0.4), scale=(0.5,1.5), shear=None, resample=False, fillcolor=0),
			transforms.ToTensor()
		])

train_dataset = datasets.ImageFolder(
		traindir,
		all_transforms)

val_dataset = datasets.ImageFolder(
		valdir,
		val_transforms)



model = models.shufflenet_v2_x2_0(pretrained=False)
model.fc = nn.Linear(2048,5)


device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model = nn.DataParallel(model)

if sys.argv[1] == "load":
	model.load_state_dict(torch.load(sys.argv[2]))

train_loader = torch.utils.data.DataLoader(
	train_dataset, 
	batch_size=128, 
	shuffle=True,
	num_workers=2)

val_loader = torch.utils.data.DataLoader(
	val_dataset, 
	batch_size=128, 
	shuffle=False,
	num_workers=2)



l_rate=0.001

optimizer = optim.Adam(model.parameters(), lr=l_rate,weight_decay=0.00001)
criterion = nn.CrossEntropyLoss()

ep = 60

for x in range(ep):

	losses = AverageMeter('Loss', ':.6f')

	for i, (inputs, label) in enumerate(train_loader):

		inputs, label = inputs.to(device), label.to(device)

		output = model(inputs)
		output = F.softmax(output, dim = 1)
		
		loss = criterion(output, label)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		losses.update(loss.item(), inputs.size(0))

		# print("\tStep: ",i,losses.__str__())

	print("Epoch: ",x,losses.__str__())
	validate(val_loader,model,criterion,device)
	torch.save(model.state_dict(), "./models/shfl-1")
	# resnet50 saved as r18-1 here !



