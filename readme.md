
# Game of Deep Learning - Approach Document

### By Vedant Bassi

This competition is based on image classification on the ship images dataset provided by Analytics Vidya.

There are 5 classes overall.

```'Cargo': 1,  'Military': 2,  'Carrier': 3, 'Cruise': 4,  'Tankers': 5```

Cargo has 2120 images
Military has 1167 images
Carrier has 916 images
Cruise has 832 images
Tanker has 1217 images

Totally : 6252 images

Data Split -  (88% in train 12% in validation)
5500 images in train
752 images in validation
(As seen in the makeFolders.py file)


My Approach:
1. Use state of the art models pre-trained on Imagenet (Transfer Learning)
2. Use ensemble methods to aggregate predictions from different models to boost  performance
3. Use Curriculum Learning with cyclical learning rate to finetune models and increase model accuracy

List of Models used:

ResNet 152 		(https://arxiv.org/abs/1512.03385)  
PNAS Net 5 Large	(https://arxiv.org/abs/1712.00559)  
ResNeXt-101 		(https://arxiv.org/abs/1611.05431)  
EfficentNet B3 	(https://arxiv.org/abs/1905.11946)  

All of the above models were pre-trained on ImageNet.
The final Model is was an ensemble of the above 4 models.


## Training Environment
Used Pytorch to train all my models and ran the code on a Deep Learning Server on Google Cloud.
All trainings were done on a single Nvidia K80 GPU, with batch sizes varying from 32 to 64
Depending on the size of the model.

## Validation metrics
Used the f1_score() and accuracy_score() from the sklearn.metrics module to get the weighted FI score on the validation set.

## Cyclical Learning Rate Schedule
The learning rate schedule was based on Super Convergence , a concept taken from this paper: https://arxiv.org/abs/1708.07120

With this approach, the learning rate is slowly increased and then reaches a peak, and then keeps decreasing to a smaller value.

## Pretrained Models & Where to find them
The PNAS NET model was downloaded from the Cadene repository (https://github.com/Cadene/pretrained-models.pytorch#pnasnet)

The ResNeXt and ResNet152 models was downloaded from the official Torchvision models repository.
(https://github.com/pytorch/vision)

The model EfficentNet B3 was downloaded from the  Luke Melas repository
(https://github.com/lukemelas/EfficientNet-PyTorch)


## Custom MLP Model Architecture

All pretrained models had a final output layer of size 1000. Did not freeze the pretrained models.
This was followed by a custom 5 layer MLP designed by me.

Each layer was followed by a Batch Normalization layer
Applied Parametric ReLU on each layer except the last two layers
Input ~> 1024 ~> 512 ~> 256 ~> 128 ~> 5 ~> Softmax

## Optimizer and Loss Function
Used Adam optimizer with starting learning rate of 0.00001 and then increasing up to 0.00012 with increments of x2 every 15 epochs then reducing it to 0.000001 over several epochs.

Weight decay was set to 0.0001

Used the nn.CrossEntropyLoss() function.

Trained each model to 80 epochs and used early stopping to avoid overfitting.

Ran training on the entire train set (of 6252 images) for 3/4 epochs to boost model performance.

The optimizer runs backward pass on both the Pre-Trained model and the MLP model in every step.


## Data Augmentation and Preprocessing

Used the built-in Pytorch torchvision.transforms functions for Data Augmentation.

Here they are :
```
validation_transforms = transforms.Compose([
	transforms.Resize((331,331)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
		])


Training_transforms = transforms.Compose([
	transforms.Resize((331,331)),
	transforms.RandomHorizontalFlip(),
	transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1, hue=0.1),
	transforms.RandomAffine(degrees=15, translate=(0.3,0.3), scale=(0.5,1.5), shear=None, resample=False, fillcolor=0),
	transforms.ToTensor(),
	transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])	
		])
```


Normalized all inputs with mean  = [0.5 0.5 0.5] and standard deviation = [0.5 0.5 0.5]
This helped in better model training, probably because the pretrained models were also trained with normalized inputs.



## Ensembling Predictions

The outputs the final layer from each model were taken and added.
Then applied an argmax to find the index of the max value and that was considered as the class for the final prediction.


