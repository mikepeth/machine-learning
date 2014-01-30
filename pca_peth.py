#!/usr/bin/env python

"""
My own personal PCA code
"""

from numpy import *
#from pylab import *
import os
import pyfits
from pygoods import *
#from scipy import *

def whiten(data):
    mu = mean(data,axis=0)
    wvar = std(data,axis=0)

    whiten_data = zeros(shape(data))
    for p in range(len(mu)):
        whiten_data[:,p]  = (data[:,p] - mu[p])/wvar[p]
    return whiten_data
    

def PCA(whiten_data,covariance=False):
    #number of objects x number of features
    if covariance == False:
        u, s, v = linalg.svd(whiten_data)
        eigenvalues = s**2/sum(s**2)

        pc = zeros(shape(whiten_data))
        for i in range(len(whiten_data[0])):
                       for j in range(len(whiten_data[0])):
                                      pc[:,i] = pc[:,i] + v[i][j]*whiten_data[:,j]
                                      
        return pc,v,eigenvalues
    
    if covariance == True:
        C = cov(whiten_data) #Covariance matrix
        l,v = linalg.eig(C)
        l_sort = sorted(l,reverse=True)
        ev = array(l_sort)**2/sum(array(l_sort)**2)
        x = v.transpose()
        y=x[l.argsort()]
        eigenvector_sort = y[::-1]
        z = dot(eigenvector_sort.T,whiten_data)
        return z,eigenvector_sort,ev

def PCA_z(whiten_data,v,covariance=False):
    #number of objects x number of features
    if covariance == False:
        pc = zeros(shape(whiten_data))
        for i in range(len(whiten_data[0])):
            for j in range(len(whiten_data[0])):
                pc[:,i] = pc[:,i] + v[i][j]*whiten_data[:,j]

    if covariance == True:
        z = dot(v,whiten_data)
    
    return pc
