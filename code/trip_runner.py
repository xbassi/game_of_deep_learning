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


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()



def get_triplets(inputs,labels):

	a = []
	p = []
	n = []
	for i in range(labels.shape[0]):
		
		fp = 0
		fn = 0
		a.append(inputs[i])

		for j in range(labels.shape[0]):

			if i == j:
				continue
			elif labels[i] == labels[j] and fp == 0:
				p.append(inputs[j])
				fp = 1
			elif labels[i] != labels[j] and fn == 0:
				n.append(inputs[j])
				fn = 1
			elif fn == 1 and fp == 1:
				break

		if fp == 0:
			p.append(inputs[i])

	if len(a) == len(p) and len(a) == len(n):

		

		a = torch.stack(a)
		p = torch.stack(p)
		n = torch.stack(n)

		return a,p,n

	else:
		print('Error: Triplet Lengths did not match.')
		exit()


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
			transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.8,1.2), shear=None, resample=False, fillcolor=0),
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


model = models.resnet50(pretrained=True)
# model.fc = nn.Linear(2048,512)

# resnext101_32x8d
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
	batch_size=80, 
	shuffle=True,
	num_workers=2)

val_loader = torch.utils.data.DataLoader(
	val_dataset, 
	batch_size=80,
	shuffle=False,
	num_workers=2)



l_rate=0.00001
optimizer = optim.Adam(
	list(model.parameters()) + 
	list(mlp.parameters()), lr=l_rate,weight_decay=0.00001)



# optimizer = optim.SGD(list(model.parameters()) + 
# 	list(mlp.parameters()), lr=0.1, momentum=0.9)

criterion = nn.CrossEntropyLoss()
tl = TripletLoss(5.0)


ep = 100

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# model.eval()

for x in range(1,ep):

	losses = AverageMeter('XEntropy Loss', ':.6f')
	tlosses = AverageMeter('Triplet Loss', ':.6f')
	# scheduler.step()
	k = -1
	for i, (inputs, label) in enumerate(train_loader):

		
		inputs, label = inputs.to(device), label.to(device)

		output = model(inputs)

		a,p,n = get_triplets(output,label)

		output = mlp(output)
		output = F.softmax(output, dim = 1)
		

		

		tl_loss = tl(a,p,n)

		loss = criterion(output, label)

		total_loss = loss + tl_loss

		optimizer.zero_grad()
		total_loss.backward()
		optimizer.step()


		losses.update(loss.item(), inputs.size(0))
		tlosses.update(tl_loss.item(), inputs.size(0))

		k += 1
		if k % 10 == 0:
			print("\tStep: ",i,losses.__str__())
			print("\t",tlosses.__str__())

	print("\nEpoch: ",x,losses.__str__())
	validate(val_loader,model,mlp,criterion,device)
	# validate(train_loader,model,mlp,criterion,device)
	torch.save(model.state_dict(), "./models/r50-30")
	torch.save(mlp.state_dict(), "./models/mlp-30")



