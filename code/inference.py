import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image

import skimage
import skimage.io as io
import skimage.transform as skt
import pandas as pd 
import numpy as np

from custom_model import MLP


import pretrainedmodels

transform = transforms.Compose([
			transforms.Resize((224,224)),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

transform = transforms.Compose([
			transforms.Resize((331,331)),
			# transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])




test = pd.read_csv("test.csv")



# model = models.resnext101_32x8d(pretrained=True)
# model.fc = nn.Linear(2048,5)


model_name = 'pnasnet5large' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')

# mlp = MLP(model.fc.out_features)

mlp = MLP(1000)

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model = nn.DataParallel(model)

model.load_state_dict(torch.load("./models/pnasnet5large-28"))
model.eval()


mlp.to(device)
mlp = nn.DataParallel(mlp)
mlp.load_state_dict(torch.load("./models/mlp-28"))
mlp.eval()

batch = []
rid = []
k = None
c = 0
for row in test.values:

	img = Image.open("./images/"+row[0]).convert("RGB")
	# img = skt.resize(img,(224,224))

	# if len(img.shape) == 2:
	# 	img = np.stack((img,)*3, axis=-1)

	img = transform(img)
	
	batch.append(img)
	rid.append(row[0])

	if len(batch) > 4:
		c += 1
		print(c*10)

		if k is None:
			batch = torch.stack(batch)
			batch = model(batch.float())
			batch = mlp(batch)
			batch = F.softmax(batch, dim = 1)
			batch = batch.detach().cpu().numpy()
			k = np.argmax(batch, axis=1)
			k = k + 1

		else:
			batch = torch.stack(batch)
			batch = model(batch.float())
			batch = mlp(batch)
			batch = F.softmax(batch, dim = 1)
			batch = batch.detach().cpu().numpy()
			batch = np.argmax(batch, axis=1)
			batch = batch + 1
			k = np.concatenate((k,batch) , axis=0)
		batch = []

if len(batch) != 0:
	batch = torch.stack(batch)
	batch = model(batch.float())
	batch = mlp(batch)
	batch = F.softmax(batch, dim = 1)
	batch = batch.detach().cpu().numpy()
	batch = np.argmax(batch, axis=1)
	batch = batch + 1
	k = np.concatenate((k,batch) , axis=0)

submission = np.vstack((rid,k)).T

np.savetxt("pnasnet5large-mlp-28.csv",
	submission, 
	delimiter=",",
	fmt="%s",
	header="image,category",
	comments='')
