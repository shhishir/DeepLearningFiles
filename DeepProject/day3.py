#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 15:04:51 2020

@author: shishir
"""

from libcpab.pytorch import cpab
import matplotlib.pyplot as plt
import numpy as np
import torch
from keras.datasets import mnist

#loading two sets of images using keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_seven = x_train[y_train == 8]
imgs = x_train_seven[0:2, :, :]

# Number of transformed samples 
N = 1
numImg = len(imgs)
thetas = np.zeros((10 * (numImg-1),34))
row = 0


#%%
for i in range(numImg):
    data1 = x_train_seven[i]
    data1 = np.tile(np.expand_dims(data1, 2), [N,1,1,1]) 
    print(np.shape(data1))
    data1 = torch.Tensor(data1).permute(0,3,1,2)
    data1 = np.reshape(data1, (N,1,28,28))
    
    for j in range(numImg):
        if i==j:
            continue
    
        data2 = x_train_seven[j]   
        data2 = np.tile(np.expand_dims(data2, 2), [N,1,1,1])    
        print(np.shape(data2))
        data2 = torch.Tensor(data2).permute(0,3,1,2)
        data2 = np.reshape(data2, (N,1,28,28))
        
        T2 = cpab(tess_size=[3,3], device='cpu')
        theta_est = T2.identity(1, epsilon=1e-4)
        theta_est.requires_grad = True
        
        optimizer = torch.optim.Adam([theta_est], lr=0.01)
    
        # Optimization loop
        loop = 100
    
        for z in range(loop):
            trans_est = T2.transform_data(data1, theta_est, outsize=(28, 28))
            loss = (data2.to(trans_est.device) - trans_est).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print('Iter', i, ', Loss', np.round(loss.item(), 4), 
                  ', ||theta_true - theta_est||: ',
                  np.linalg.norm((data2-trans_est.cpu().detach()).numpy().round(4)))
            
            thetas[(i+1)*j,:] = torch.Tensor.cpu(theta_est).detach().numpy()[:]
            print(np.shape(thetas))
            
            M = thetas.mean(0)

            C = np.cov(thetas, rowvar=False)
            
            theta_star = np.random.multivariate_normal(M, C, 34).T
            theta_star = torch.Tensor(theta_star)
            
            data = np.expand_dims(x_train_seven[0], 2)
            data = np.expand_dims(data, 0)
            data = torch.Tensor(data).permute(0,3,1,2)
            #data = np.reshape(data, (N,1,28,28))

            T1 = cpab(tess_size=[3,3])
            theta_true = theta_star[:1]
            
            new_data = T1.transform_data(data, theta_true, outsize=(28, 28))
            

#%%  Plotting the results

plt.subplots(1,2, figsize=(10, 15))

#Source image
plt.subplot(1,3,1)
data = np.reshape(data, (1,28,28))
plt.imshow(data.permute(0,1,2).numpy()[0])
plt.axis('off')
plt.title('Source')

"""
#Target image
plt.subplot(1,3,2)
data2 = np.reshape(data2, (1,28,28))
plt.imshow(data2.permute(0,1,2).cpu().numpy()[0])
plt.axis('off')
plt.title('Target')
"""

# Estimated image
plt.subplot(1,3,2)
new_data = torch.reshape(new_data, (1,28,28))
plt.imshow(new_data.permute(0,1,2).cpu().detach().numpy()[0])
plt.axis('off')
plt.title('New Data')
plt.show()

#print('N: ', N) 

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    