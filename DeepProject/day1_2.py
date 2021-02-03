#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 13:15:29 2020

@author: shishir
"""
#data = pd.read_csv('/Users/shishir/Documents/Deep Learning/DeepProject/archive/trainingSet/trainingSet/0/')

from libcpab.pytorch import cpab
#from libcpab.helper.utility import show_images
#import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
import numpy as np
import torch


data1 = plt.imread('/Users/shishir/Documents/Deep Learning/DeepProject/archive/trainingSet/trainingSet/0/img_10007.jpg') / 255
data2 = plt.imread('/Users/shishir/Documents/Deep Learning/DeepProject/archive/trainingSet/trainingSet/0/img_11717.jpg') / 255

#plt.imshow(data1)
#print('Sample data: ', np.shape(data1))
# Number of transformed samples 
N = 1

# Coverting to three dimention [number of sample, n_channels, width, height]
data1 = np.tile(np.expand_dims(data1, 0), [N,1,1,1])
print('Expanded sample diamention data: ', np.shape(data1))# create batch of data
# Convert to torch tensor and torch format [n_batch_size, rgb, width, height]
data1 = torch.Tensor(data1).permute(0,3,1,2)
#print('Torch size data: ',np.shape(data1))
data1 = np.reshape(data1, (N,1,28,28))
#print('Torch reshaped size data: ',np.shape(data1))


data2 = np.tile(np.expand_dims(data2, 0), [N,1,1,1])
# Convert to torch tensor and torch format [n_batch, n_channels, width, height]
data2 = torch.Tensor(data2).permute(0,3,1,2)
data2 = np.reshape(data2, (N,1,28,28))  
print(np.shape(data2))# create batch of data    

#%%
# Define transformer class
#T = cpab(tess_size=[3,3], device='cpu')
# Sample random transformation
#theta = 0.5*T.sample_transformation(N)

#T1 = cpab(tess_size=[3,3])
#theta_true = 0.5*T1.sample_transformation(1)
#transformed_data = T1.transform_data(data1, theta_true, outsize=(28, 28))

T2 = cpab(tess_size=[3,3], device='cpu')
theta_est = T2.identity(1, epsilon=1e-4)
theta_est.requires_grad = True
#transformed_data = T2.transform_data(data2, theta_est, outsize=(28, 28))
optimizer = torch.optim.Adam([theta_est], lr=0.01)
    
# Optimization loop
maxiter = 100
    
for i in range(maxiter):
    trans_est = T2.transform_data(data1, theta_est, outsize=(28, 28))
    loss = (data2.to(trans_est.device) - trans_est).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Iter', i, ', Loss', np.round(loss.item(), 4), ', ||theta_true - theta_est||: ',
          np.linalg.norm((data2-trans_est.cpu().detach()).numpy().round(4)))
    print(np.shape(theta_est))
    
    
#%%
# Show the results
plt.subplots(1,3, figsize=(10, 15))

plt.subplot(1,3,1)
data1 = np.reshape(data1, (1,28,28))
plt.imshow(data1.permute(0,1,2).numpy()[0])
plt.axis('off')
plt.title('Source')

#Target image
plt.subplot(1,3,2)
data2 = np.reshape(data2, (1,28,28))
plt.imshow(data2.permute(0,1,2).cpu().numpy()[0])
plt.axis('off')
plt.title('Target')

# Estimated image
plt.subplot(1,3,3)
trans_est = torch.reshape(trans_est, (1,28,28))
plt.imshow(trans_est.permute(0,1,2).cpu().detach().numpy()[0])
plt.axis('off')
plt.title('Estimate')
plt.show()

#%%
# Transform the images
#transformed_data = T.transform_data(data1, theta_est, outsize=(28, 28))
#print('Transformed image data size : ',np.shape(transformed_data))


#%%
# Get the corresponding numpy arrays in correct format
#transformed_data = transformed_data.permute(0, 2, 3, 1).cpu().numpy()

#print('Transformed image data size after permute : ',np.shape(transformed_data))


#transformed_data = np.reshape(transformed_data, (N,1,28,28))
#print('Transformed image data size after permute and resize : ',np.shape(transformed_data))


#%%
# Show transformed samples
#transformed_data = np.reshape(transformed_data, (N,28,28))
#show_images(transformed_data)











