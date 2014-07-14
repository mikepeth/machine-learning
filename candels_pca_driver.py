#!/usr/bin/env python
#R20=0.08
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
import convex_check as cvx
import stamps as st
import stat_tools as stat
import candels_pca_functions as cpf

    

#Global Variables
data_fields_j = ['C_J','M20_J','G_J','A_J','M_J','I_J','D_J']
data_fields_h = ['C_H','M20_H','G_H','A_H','M_H','I_H','D_H']
data_fields_i = ['C_I','M20_I','G_I','A_I','M_I','I_I','D_I']
data_fields_v = ['C_V','M20_V','G_V','A_V','M_V','I_V','D_V']
data_fields_y = ['C_Y','M20_Y','G_Y','A_Y','M_Y','I_Y','D_Y']

#Main Program
#Read-in Catalogs
#UDS Catalogs
uds_hdu = pyfits.open('/Users/Mike/Dropbox/candels/morph/uds_master_rainbow.fits') #UDS
uds = uds_hdu[1].data

cos_hdu = pyfits.open('/Users/Mike/Dropbox/candels/morph/cos_master_rainbow.fits') #COS
cos = cos_hdu[1].data

goodss_hdu = pyfits.open('/Users/Mike/Dropbox/candels/morph/gs_master_rainbow.fits') #GOODS-S
goodss = goodss_hdu[1].data

goodsn_hdu = pyfits.open('/Users/Mike/Dropbox/candels/morph/goodsn/gn_jh_master.fits') #GOODS-N, only Y-band ATM
goodsn = goodsn_hdu[1].data

candels_j = dm.data_multicat_full(uds,catalog2=goodss,catalog3=cos,band='J',zmin=1.36,zmax=1.97)
candels_y = dm.data_multicat_full(goodsn,band='Y',zmin=1.05,zmax=1.36)

#Get only data we want, only 7 Morphologies
A_j_morph = dm.data_multicat(uds,data_fields_j, catalog2=goodss,catalog3=cos,band='J',zmin=1.36,zmax=1.97)
A_h_morph = dm.data_multicat(uds,data_fields_h, catalog2=goodss,catalog3=cos,band='H',zmin=1.99,zmax=2.57)
A_i_morph = dm.data_multicat(uds,data_fields_i, catalog2=cos, band='I',zmin=0.51,zmax=1.05)
A_v_morph = dm.data_multicat(uds,data_fields_v, catalog2=cos, band='V',zmin=0.0,zmax=0.51)
A_y_morph = dm.data_multicat(goodsn,data_fields_y, band='Y',zmin=1.05,zmax=1.36)
#print "F105W", len(A_y.ra), " Galaxies"

#Calculate PCs
pc   = dm.pca_multicat(A_j_morph.morph,calculate_vectors=True)
pc_h = dm.pca_multicat(A_h_morph.morph,A_j=A_j_morph.morph,calculate_vectors=False)
pc_i = dm.pca_multicat(A_i_morph.morph,A_j=A_j_morph.morph,calculate_vectors=False)
pc_v = dm.pca_multicat(A_v_morph.morph,A_j=A_j_morph.morph,calculate_vectors=False)
pc_y = dm.pca_multicat(A_y_morph.morph,A_j=A_j_morph.morph,calculate_vectors=False)
#pc_j_fixc = dm.pca_multicat(A_j_fixc.morph,A_j=A_j_morph.morph,calculate_vectors=False)



#Clustering
clusters=10
ward = Ward(n_clusters=clusters).fit(pc.X)
label_j = ward.labels_
label_j[where((label_j == 5) | (label_j == 7) | (label_j == 8))[0]]=-1 #regroup all outliers (5,7,8) into group=-1



#Labels for F160W 2<z<2.5
## print 'Labels for F160W 2<z<2.5'
## label_h = cvx.pc_convexhull(pc_h.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

## print 'Labels for F814W 0.5<z<1.05, Doesnt include GOODS-S'
## #Labels for F814W 0.5<z<1.05, Doesn't include GOODS-S
## label_i = cvx.pc_convexhull(pc_i.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

## print 'Labels for F606W 0.0<z<0.5, Doesnt include GOODS-S'
## #Labels for F814W 0.0<z<0.5, Doesn't include GOODS-S
## label_v = cvx.pc_convexhull(pc_v.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

print 'Labels for F105W 0.5<z<1.05, GOODS-N ONLY'
#Labels for F105W 0.5<z<1.05, GOODS-N ONLY
label_y = cvx.pc_convexhull(pc_y.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

#print 'Labels for Corrected Concentration Galaxies'
#Labels for F814W 0.0<z<0.5, Doesn't include GOODS-S
#label_concentration = cvx.pc_convexhull(pc_j_fixc.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

#cpf.stamp_collection(uds_fixc,label=label_concentration,band='f125w')
#Demographics
#uds_j = dm.redshift_sample(uds,band='J',zmin=1.36,zmax=1.97)
#close('all')
#pm.vis_class(candels_j,label_j)
#cpf.quenched_fraction(candels_j,label_j)
#pm.sersic_class(candels_j,label_j)
#pm.group_histograms(candels_j,label_j)
#pm.pc_plot_parameter(pc,candels_j)
#pm.reffMass(candels_j,label_j)
#pm.ginim20(candels_j,label_j)
#pm.concentrationAsymmetry(candels_j,label_j)
#pm.iAsymmetry(candels_j,label_j)
#pm.dAsymmetry(candels_j,label_j)
#pm.mAsymmetry(candels_j,label_j)
#pm.reffq(candels_j,label_j)

#pm.vis_class(candels_y,label_y)
#cpf.quenched_fraction(candels_y,label_y)
#pm.sersic_class(candels_y,label_y)
pm.group_histograms(candels_y,label_y)
#pm.pc_plot_parameter(pc,candels_y)
pm.reffMass(candels_y,label_y)
pm.ginim20(candels_y,label_y)
pm.concentrationAsymmetry(candels_y,label_y)
pm.iAsymmetry(candels_y,label_y)
pm.dAsymmetry(candels_y,label_y)
pm.mAsymmetry(candels_y,label_y)
## cpf.cluster_plots_1field(pc_y,label=label_y,band='y')
#pm.latexClasstable(label_y,band='y')
#pm.latexClasstable(label_j,band='j')
#pm.latexClasstable(label_h,band='h')
#pm.latexClasstable(label_i,band='i')
#pm.latexClasstable(label_v,band='v')
pm.reffq(candels_y,label_y)
