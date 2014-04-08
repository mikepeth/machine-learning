#!/usr/bin/env python

"""
This script will combine the UDS Multi-wavelength catalog, Photo-z/Stellar Mass catalog and Gini-M20 morphology catalog into a single catalog that
will then be run with PCA to determine the important dimensions
'/Users/Mike/Desktop/candels/morph/uds/UDS_mw_morph_viz_pz.fits' is ALL catalogs
"""

from numpy import *
import pyfits
from pygoods import *
from scipy import *
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import Ward
import matplotlib
import pylab as pl
import mpl_toolkits.mplot3d.axes3d as p3

#My Personally created modules
import data_maker as dm
import plot_maker as pm
import machine_learning as ml
        
def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimension
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimension for which a Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    #from pyhull.delaunay import DelaunayTri
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


def pc_convexhull(data,X,clusters=10,radius=0.0):
    from matplotlib import nxutils
    """
    Ward clustering based on a Feature matrix.

    Recursively merges the pair of clusters that minimally increases
    within-cluster variance.

    The inertia matrix uses a Heapq-based representation.

    This is the structured version, that takes into account some topological
    structure between samples.

    Parameters
    ----------
    data : array, shape (n_samples, n_features)
        feature matrix  representing n_samples samples to be clustered based on clusters found for X

    X : array, shape (n_samples, n_features)
        feature matrix  representing n_samples samples to be clustered
        
    clusters : int (optional)
        Stop early the construction of the tree at n_clusters. This is
        useful to decrease computation time if the number of clusters is
        not small compared to the number of samples. In this case, the
        complete tree is not computed, thus the 'children' output is of
        limited use, and the 'parents' output should rather be used.
        This option is valid only when specifying a connectivity matrix.

    radius : float (optional)
        The search radius for determining if a point is within a convex hull

    Returns
    -------
    new_labels : 1D array, shape (n_samples)
        The cluster labels for data based upon clusters created for X
    """
    
    import scipy
    from pyhull import convex_hull
    from scipy import spatial
    from matplotlib import path
    
    ward = Ward(n_clusters=clusters).fit(X)
    label = ward.labels_

    #Calculate distance between all galaxies in groups, so we can normalize later
    #multi_labels=0
     
    #Convex Hull
    #from scipy.spatial import ConvexHull

    group_label = zeros(int(len(data)))-1
    distance_sq = zeros(clusters)
    for ngal in range(len(data)):
        group = zeros(clusters)
        if ((data[ngal,0] < -8.2) | (data[ngal,0] > 3.2) | (data[ngal,1] < -3.4) | (data[ngal,1] > 4.3) | (data[ngal,2] < -2.5) | (data[ngal,2] > 3.3)):
                group_label[ngal] = -1 #outliers are put into group =-1, replaces groups 6, 7 and 8
                print "Galaxy ", ngal, " is an outlier"
                continue
        else:
            for n in range(clusters):
                labeln = where(label == n)[0]
                

                if len(labeln) > 8:
                    for pcx in range(shape(X)[1]):
                        for pcy in range(shape(X)[1]):
                            if pcy > pcx:
                                group1_pt_x,group1_pt_y  = data[ngal][pcx], data[ngal][pcy]
                                points = zeros((len(labeln),2))
                                points[:,0] = X[labeln][:,pcx]#Pick galaxies only for a specific group
                                points[:,1] = X[labeln][:,pcy]
                                hull = spatial.ConvexHull(points)

                                #Does Polygon (2D convex hull) contain point?
                                inside_poly_shape = path.Path(points[hull.vertices]).contains_point((group1_pt_x,group1_pt_y),radius=radius)
                                #inside_poly_shape = in_hull((group1_pt_x,group1_pt_y),points)
                                if inside_poly_shape ==True:
                                    group[n] = group[n]+1
                else:
                    pass

            group_number = where(group == max(group))[0]
            #print group
            #print group_number

            
            if len(group_number) > 1:
                var_new = zeros(len(group_number))
                #print group
                #print "Galaxy ", ngal
                #print group
                
                #multi_labels=#multi_labels+1 #counter for how many galaxies fall into multiple convex hulls
                distance_sq = zeros(len(group_number))
                for nn in range(len(group_number)):
                    eq_label = where(label == group_number[nn])

                    galaxy = X[ngal,:].reshape((1,len(X[0])))
                    new_points = concatenate((X[eq_label],galaxy))
      
                    for sample_ngal in range(len(eq_label)):
                        distance_sq[nn] = distance_sq[nn]+scipy.spatial.distance.sqeuclidean(data[ngal],X[eq_label][sample_ngal])

                    #var_new[nn] = sum(var(new_points))
                #distance_sq = var_new/var_tot[group_number] #Normalize variance
                if len(where(distance_sq == min(distance_sq))[0]) == 1:
                    group_label[ngal] = group_number[where(distance_sq == min(distance_sq))[0]]
                else:
                    "Galaxy ", ngal, " apparently is either really in multiple groups or none"
                    group_label[ngal] = -2

            if len(group_number) == 1:
                group_label[ngal] = group_number[0]
    
    return array(group_label,'i')
