#!/usr/bin/env python

"""
My own personal Machine Learning codes.

Includes PCA, Diffusion Mapping, Isomapping
"""

from numpy import *
#from pylab import *
import os
import pyfits
from pygoods import *
#from scipy import *

def whiten(data):
    #Take data (Nxk) N objects by k features, subtract mean, and scale by variance
    mu = mean(data,axis=0)
    wvar = std(data,axis=0)

    whiten_data = zeros(shape(data))
    for p in range(len(mu)):
        whiten_data[:,p]  = (data[:,p] - mu[p])/wvar[p]
    return whiten_data
    

def PCA(whiten_data,covariance=False):
    #whiten data is (Nxk): number of objects x number of features
    #Covariance = false uses SVD method to calculate eigenvalues, eigenvectors
    if covariance == False:
        u, s, v = linalg.svd(whiten_data)
        eigenvalues = s**2/sum(s**2)

        pc = zeros(shape(whiten_data))
        for i in range(len(whiten_data[0])):
                       for j in range(len(whiten_data[0])):
                                      pc[:,i] = pc[:,i] + v[i][j]*whiten_data[:,j]
                                      
        return pc,v,eigenvalues
    
    if covariance == True:
    #Use covariance method, take eigenvalues of C matrix
        C = cov(whiten_data) #Covariance matrix
        l,v = linalg.eig(C)
        l_sort = sorted(l,reverse=True)
        ev = array(l_sort)**2/sum(array(l_sort)**2)
        x = v.transpose()
        y=x[l.argsort()]
        eigenvector_sort = y[::-1]
        z = dot(eigenvector_sort.T,whiten_data)
        return z,eigenvector_sort,ev

def diffusionMap(data, epsilon=0.2):
    #Lee & Freeman 2012
    #data is (Nxk): N objects by k features, not whitened
    #epsilon is value determined to optimize function, default is 0.2
    distance = zeros((len(data),len(data))) #NxN distance function

    #Step 1. Build matrix (NxN) of distances using e^(-d_ij^2/epsilon), d_ij = Euclidean distance
    for i in range(len(data)): #N objects
        for j in range(len(data)): #N-i objects
            distance_sq = 0
            for k in range(len(data[0])): #k features
                distance_sq += (data[i][k] - data[j][k])**2
            distance[i][j] = math.exp(-1.0*distance_sq/epsilon)
            
    #return distance
    #Step 2. Eigendecompose distance matrix
    
    l,v = linalg.eig(distance)
    l_sort = sorted(l,reverse=True)
    ev = array(l_sort)**2/sum(array(l_sort)**2)
    x = v.transpose()
    y=x[l.argsort()]
    eigenvector_sort = y[::-1]

    #Step 3. Project original data onto newly defined basis
    z = dot(eigenvector_sort.T,data)
    
    return z,eigenvector_sort,ev #Returns scores, eigenvectors, and eigenvalues


            
            
            
            