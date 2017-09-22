# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 07:10:09 2017

@author: shrey
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pca import do_pca


#read the data
data=pd.read_csv("/home/shrey/Desktop/ml/dataset_1.csv")

#fetch the three variables
x=data['x']
y=data['y']
z=data['z']

#convert them into numpy arrays 
x_var=np.array(x)
y_var=np.array(y)
z_var=np.array(z)

#Calculate Variance
variance_x=np.var(x)
variance_y=np.var(y)
variance_z=np.var(z)

#Calculate the covariance
cov_x_y=np.cov(x,y)
cov_y_z=np.cov(y,z)

# Now to calcualte the new transformed matrix y and principal component matriz
dataset=np.array(data)
n=2 #number of principal components I need
y,m=do_pca(dataset,n) # do_pca is my function I made in my module

print(y)  #y is the new transformed matrix
print(m)  #m is the principal component matrix having 2 principal components
