# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:03:53 2017

@author: shrey
"""

import numpy as np


def do_pca(data,n):


    dataset=np.array(data)
    
    mean=dataset.mean(axis=0)
    
    data_mc=dataset-mean
    
    #data_cov=np.cov(data_mc[:,0],data_mc[:1],data_mc[:2])
    
    data_cov=np.cov(data_mc.T)
    
    e_val,e_vec=np.linalg.eig(data_cov)
    
    eig_pairs = [(np.abs(e_val[i]), e_vec[:,i]) for i in range(len(e_val))]
    
    eig_pairs.sort()
    eig_pairs.reverse()
    
    matrix_w = np.hstack((eig_pairs[i][1].reshape(dataset.shape[1],1)) for i in range(n))    
    
    
    #mat_w = np.hstack(matrix_w) 
                            
    print(matrix_w)                     
    y=data_mc.dot(matrix_w) 
    return y,matrix_w 