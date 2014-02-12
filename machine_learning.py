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
import scipy
from scipy.sparse.linalg import eigsh

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

def diffusionMap(data, epsilon=0.2,delta=1e-10,n_eig=100):
    #Lee & Freeman 2012
    #data: (Nxk): N objects by k features, not whitened
    #epsilon: value determined to optimize function, default is 0.2
    #delta: minimum value used in e^(-d^2/eps) matrix, creates sparse matrix
    #n_eig: Number of eigenvectors to keep, default is 100 
    distance = zeros((len(data),len(data))) #NxN distance function

    #Step 1. Build matrix (NxN) of distances using e^(-d_ij^2/epsilon), d_ij = Euclidean distance
    for i in range(len(data)): #N objects
        for j in range(len(data)): #N-i objects
            distance_sq = scipy.spatial.distance.euclidean(data[i],data[j])*scipy.spatial.distance.euclidean(data[i],data[j])
            distance[i][j] = math.exp(-1.0*distance_sq/epsilon)
            if distance[i][j] < delta:
                distance[i][j] = 0.0 #Removes values that are very small

    k = sqrt(sum(distance,axis=1)).reshape((len(distance),1)) #sqrt of the sum of the rows
    A = distance/(inner(k,k))
    
    #Step 2. Eigendecompose distance matrix
    N = shape(data)[0]
    l, v = linalg.eig(A)

    #Sort eigenvectors based on eigenvalues
    l_sort = sorted(l,reverse=True)
    sort_indx = argsort(l)[::-1]
    
    #Place sorted eigenvectors into new matrix
    eigenvector_sort = zeros(shape(v))
    for indx in range(len(sort_indx)):
        eigenvector_sort[indx] = v[sort_indx][indx]

    #Create matrix containing eigenvalues
    eigenvalues = array((l_sort[1:])).reshape(len(l_sort[1:]),1)
    lambda_x = inner(ones((N,1)),eigenvalues)

    #Psi = Eigenvectors/eigenvector[1]
    #print shape(eigenvector_sort[0])
    weight = inner(eigenvector_sort[:,0].reshape(N,1),ones((N,1)))
    psi = eigenvector_sort #/weight #[:,n_eig]
    
    #Step 3. Project original data onto newly defined basis
    X = psi[:,1:n_eig+1]*lambda_x[:,0:n_eig]

    X_dict = {'X':X,'eigenvals':l_sort,'psi':psi,'weight':weight}
    return X_dict #Returns scores and eigenvalues
            
            
