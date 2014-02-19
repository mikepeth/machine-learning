#!/usr/bin/env python

"""
My own personal Machine Learning codes.

Includes PCA, Diffusion Mapping, Isomapping
"""

__author__ = "Michael Peth, Peter Freeman"
__copyright__ = "Copyright 2014"
__credits__ = ["Michael Peth", "Peter Freeman"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Michael Peth"
__email__ = "mikepeth@pha.jhu.edu"

from numpy import *
import os
import pyfits
from pygoods import *
import scipy
from scipy.sparse.linalg import eigsh

def whiten(data):
    '''
    Compute a Principal Component analysis p for a data set

    Parameters
    ----------
    data: matrix
    Input data (Nxk): N objects by k features, whitened (i.e. average subtracted and stddev scaled)


    Returns
    -------
    whiten_data: matrix
    Input data that has been average subtracted and stddev scaled (Nxk)
    '''
    
    #Take data (Nxk) N objects by k features, subtract mean, and scale by variance
    mu = mean(data,axis=0)
    wvar = std(data,axis=0)

    whiten_data = zeros(shape(data))
    for p in range(len(mu)):
        whiten_data[:,p]  = (data[:,p] - mu[p])/wvar[p]
    return whiten_data
    

def PCA(self, data):
     '''
    Compute a Principal Component analysis p for a data set

    Parameters
    ----------
    whiten_data: matrix
    Input data (Nxk): N objects by k features


    Returns
    -------
    Structure with the following keys:
    
    pc: matrix
    Principal Component Coordinates

    values: array
    Eigenvalue solutions to SVD

    vectors: matrix
    Eigenvector solutions to SVD
    '''

    #Whiten data (i.e. average subtracted and stddev scaled)
    whiten_data = whiten(data)

    #Calculate eigenvalues/eigenvectors from SVD
    u, s, v = linalg.svd(whiten_data)

    #Force eigenvalues between 0 and 1
    eigenvalues = s**2/sum(s**2)

    #Change data to PC basis
    pc = zeros(shape(whiten_data))
    for i in range(len(whiten_data[0])):
        for j in range(len(whiten_data[0])):
            pc[:,i] = pc[:,i] + v[i][j]*whiten_data[:,j]

    self.pc = pc
    self.values = eigenvalues
    self.vectors = v

    return
                                      


def diffusionMap(self, data, epsilon=0.2,delta=1e-10,n_eig=100):
    '''
    Compute a diffusion Map for a data set, based heavily on
    Lee & Freeman 2012

    Parameters
    ----------
    data: matrix
    Input data (Nxk): N objects by k features, not whitened

    epsilon: float
    value determined to optimize function, default is 0.2

    delta: float
    delta: minimum value used in e^(-d^2/eps) matrix, creates sparse matrix

    n_eig: int
    Number of eigenvectors to keep, default is 100

    Returns
    -------
    Structure with the following keys:
    
    X: matrix
    Diffusion Map Coordinates = weighted eigenvectors * eigenvalues

    eigenvals: array
    Eigenvalue solutions to Diffusion problem

    psi: matrix
    Weighted eigenvectors

    weights: array
    First eigenvector
    '''
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
    weight = inner(eigenvector_sort[:,0].reshape(N,1),ones((N,1)))
    psi = eigenvector_sort/weight
    
    #Step 3. Project original data onto newly defined basis
    X = psi[:,1:n_eig+1]*lambda_x[:,0:n_eig]

    self.X = X
    self.eigenvals = l_sort
    self.psi = psi
    self.weight = weight

    return
            
            
