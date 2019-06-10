import os
import sys
import pandas as pd 
import numpy as np 

cmd = 'rm -rf t_classes'
os.system(cmd)

cmd = 'mkdir t_classes'
os.system(cmd)

os.makedirs('t_classes/1')
os.makedirs('t_classes/2')
os.makedirs('t_classes/3')
os.makedirs('t_classes/4')
os.makedirs('t_classes/5')

cmd = 'rm -rf v_classes'
os.system(cmd)

cmd = 'mkdir v_classes'
os.system(cmd)

os.makedirs('v_classes/1')
os.makedirs('v_classes/2')
os.makedirs('v_classes/3')
os.makedirs('v_classes/4')
os.makedirs('v_classes/5')


data = pd.read_csv("train.csv").values
print("Before: ",data.shape)

# np.random.shuffle(data)

train = data[:5500]
val = data[5500:]


print("After t: ",train.shape)
print("After v: ",val.shape)

for x in train:
	# print(x)
	cmd = "cp ./images/"+x[0]+" ./t_classes/"+str(x[1])+"/"
	# cmd = "cp ./images/"+x[0]+" ./i_classes/"
	
	os.system(cmd)
	
for x in val:
	# print(x)
	cmd = "cp ./images/"+x[0]+" ./v_classes/"+str(x[1])+"/"
	# cmd = "cp ./images/"+x[0]+" ./i_classes/"
	
	os.system(cmd)


cmd = 'rm -rf pytorch-image-models/train'
os.system(cmd)

cmd = 'rm -rf pytorch-image-models/validation'
os.system(cmd)

cmd = 'cp -r t_classes pytorch-image-models/train'
os.system(cmd)

cmd = 'cp -r v_classes pytorch-image-models/validation'
os.system(cmd)


