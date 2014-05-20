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
import convex_check as cvx
import stamps as st
import stat_tools as stat

    

#Global Variables
data_fields_j = ['C_J','M20_J','G_J','A_J','M_J','I_J','D_J']
data_fields_h = ['C_H','M20_H','G_H','A_H','M_H','I_H','D_H']
data_fields_i = ['C_I','M20_I','G_I','A_I','M_I','I_I','D_I']
data_fields_v = ['C_V','M20_V','G_V','A_V','M_V','I_V','D_V']
data_fields_y = ['C_Y','M20_Y','G_Y','A_Y','M_Y','I_Y','D_Y']

#bands = ['Vband','Iband','Jband','Hband']
#redshift_range_v = [0.0,0.5]
#redshift_range_i = [0.5,1.05]
#redshift_range_j = [1.36,1.97]
#redshift_range_h = [2.0,2.57]
#scatter_parameters = ['mass','sfr','m20','photz','gini','disk','irr','mbrf','c','d','merger']

#Main Program
#Read-in Catalogs
#UDS Catalogs
uds_hdu = pyfits.open('/Users/Mike/Dropbox/candels/morph/uds_master_rainbow.fits') #UDS
uds = uds_hdu[1].data

cos_hdu = pyfits.open('/Users/Mike/Dropbox/candels/morph/cos_master_rainbow.fits') #COS
cos = cos_hdu[1].data

goodss_hdu = pyfits.open('/Users/Mike/Dropbox/candels/morph/gs_master_rainbow.fits') #GOODS-S
goodss = goodss_hdu[1].data

#Error bars from UDF, GOODS-S comparison
data_fields_h_udf = data_fields_h + ['Rpet_ell_H']+ ['SN_H']
data_fields_j_udf = data_fields_j + ['Rpet_ell_J']+ ['SN_J']

udf_h = dm.udf_errorbars(goodss,data_fields_h_udf,band='H')
udf_j = dm.udf_errorbars(goodss,data_fields_j_udf,band='J')

#Bootstrapping to get average error bars
#ggood = where((udf_j.g_error > -5) & (udf_j.g_error < 5))[0]
#g_bootstrap = stat.bootstrap(udf_j.g_error[ggood],N_iter=10000)

#print "Bootstrapping Gini Coefficient"
#g_bootstrap_by_mag = stat.bootstrap_by_mag(udf_j.mag, udf_j.g_error)

#Graphs showing error bars
#pm.morph_error(udf_j,band='J')
#pm.morph_error_new(goodss,udf_h,band='H')
#pm.morph_error_bootstrap(udf_h,band='H')
#pm.morph_error_bootstrap_percent(goodss,udf_h,band='H')

#pm.morph_error_bootstrap(udf_j,band='J')



#simple_demographics(uds,field='UDS')
#simple_demographics(cos,field='COSMOS')
#simple_demographics(goodss,field='GOODS-S')


#Keep and combine entire data dictionary for all fields
A_j = dm.data_multicat2(uds,catalog2=goodss,catalog3=cos,band='J',zmin=1.36,zmax=1.97)

#Get only data we want, only 7 Morphologies
A_j_morph = dm.data_multicat(uds,data_fields_j, catalog2=goodss,catalog3=cos,band='J',zmin=1.36,zmax=1.97)
A_h_morph = dm.data_multicat(uds,data_fields_h, catalog2=goodss,catalog3=cos,band='H',zmin=1.99,zmax=2.57)
A_i_morph = dm.data_multicat(uds,data_fields_i, catalog2=cos, band='I',zmin=0.51,zmax=1.05)
A_v_morph = dm.data_multicat(uds,data_fields_v, catalog2=cos, band='V',zmin=0.0,zmax=0.51)
#A_y = dm.data_multicat(goodss,data_fields_y, band='Y') #,zmin=1.05,zmax=1.36)
#print "F105W", len(A_y.ra), " Galaxies"

#Calculate PCs
pc   = dm.pca_multicat(A_j_morph.morph,calculate_vectors=True)
pc_h = dm.pca_multicat(A_h_morph.morph,A_j=A_j_morph.morph,calculate_vectors=False)
pc_i = dm.pca_multicat(A_i_morph.morph,A_j=A_j_morph.morph,calculate_vectors=False)
pc_v = dm.pca_multicat(A_v_morph.morph,A_j=A_j_morph.morph,calculate_vectors=False)
#pc_y = dm.pca_multicat(A_y.morph,A_j=A_j.morph,calculate_vectors=False)

#Monte Carlo Simulations of data, including PCs
#morph_stddev_j = stat.morph_error_by_mag(udf_j)
#morph_MAD_j = stat.mad_by_mag(goodss,udf_j,band='J')
#pc_j_mc = stat.montecarlo(A_j,morph_MAD_j)
#Data table of PCs (with Error estimates!)
#pm.latex_table(pc.vectors.T,std(abs(pc_j_mc.vectors_mc),axis=2).T)
#print mean(abs(pc_j_mc.vectors_mc),axis=2).T
#pm.latex_table(mean(abs(pc_j_mc.vectors_mc),axis=2).T,std(abs(pc_j_mc.vectors_mc),axis=2).T)

#Clustering
clusters=10
ward = Ward(n_clusters=clusters).fit(pc.X)
label_j = ward.labels_
label_j[where((label_j == 5) | (label_j == 7) | (label_j == 8))[0]]=-1 #regroup all outliers (5,7,8) into group=-1

#Demographics
#uds_j = dm.redshift_sample(uds,band='J',zmin=1.36,zmax=1.97)
close('all')
pm.vis_class(A_j,label_j)
pm.sersic_class(A_j,label_j)
pm.pc_plot_parameter(pc,A_j)

#Labels for F160W 2<z<2.5
#print 'Labels for F160W 2<z<2.5'
#label_h = cvx.pc_convexhull(pc_h.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

#print 'Labels for F814W 0.5<z<1.05, Doesnt include GOODS-S'
#Labels for F814W 0.5<z<1.05, Doesn't include GOODS-S
#label_i = cvx.pc_convexhull(pc_i.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

#print 'Labels for F606W 0.0<z<0.5, Doesnt include GOODS-S'
#Labels for F814W 0.0<z<0.5, Doesn't include GOODS-S
#label_v = cvx.pc_convexhull(pc_v.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

#UVJ Diagrams
#uvj_diagram(A_j,label_j)
#uvj_diagram(A_h,label_h,band='f160w')
#uvj_diagram(A_i,label_i,band='f814w')
#uvj_diagram(A_v,label_v,band='f606w')

#Test labels
#new_label = cvx.pc_convexhull(pc.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels
#print len(where(label_j != new_label)[0])/float(len(label_j))*100, " % Label Error"
#bad_label = where(label_j != new_label)[0]

#UDS Stamps & UDS Catalogs
#filename = './plots/stamp/catalogs/candels_pc_nclust10_f125w.cat'
#full_stamp_catalog(A_j,pc.X, filename,label=label_j)
#filename = './plots/stamp/catalogs/candels_pc_nclust10_f160w.cat'
#full_stamp_catalog(A_h,pc_h.X,filename,label=label_h)
#filename = './plots/stamp/catalogs/candels_pc_nclust10_f814w.cat'
#full_stamp_catalog(A_i,pc_i.X,filename,label=label_i)
#filename = './plots/stamp/catalogs/candels_pc_nclust10_f606w.cat'
#full_stamp_catalog(A_v,pc_v.X,filename,label=label_v)

#Stamps
#stamp_collection(A_j,label=label_j,band='f125w')
#stamp_collection(A_i,label=label_i,band='f814w')
#stamp_collection(A_h,label=label_h,band='f160w')

#print "F125W 1.36<z<1.97"
#simple_demographics(label_j)
#print "F160W 2<z<2.5"
#simple_demographics(label_h)
#print "F814W 0.5<z<1.0"
#simple_demographics(label_i)
#print "F125W 0.0<z<0.5"
#simple_demographics(label_v)

#Figures
#F125W
#cluster_plots_1field(A_j,pc.X,label=label_j,n=10,band='j',ssfr_check=True)

#F160W
#cluster_plots_1field(A_h,pc_h.X,label=label_h,n=10,band='h',ssfr_check=True)
#cluster_plots_ssfr_2fields(A_j,pc.X,A_h,pc_h.X,label1=label_j,label2=label_h,bands='jh',second_band='F160W')

#F814W
#cluster_plots_1field(A_i,pc_i.X,label=label_i,n=10,band='i',ssfr_check=True)
#cluster_plots_ssfr_2fields(A_j,pc.X,A_i,pc_i.X,label1=label_j,label2=label_i,bands='ji')

#F606W
#cluster_plots_1field(A_v,pc_v.X,label=label_v,n=10,band='v',ssfr_check=True)
#cluster_plots_ssfr_2fields(A_j,pc.X,A_v,pc_v.X,label1=label_j,label2=label_v,bands='jv',second_band='F606W')
#close('all')

#pelaton= where((label_j != 6) & (label_j != 7) & (label_j != 8))[0]

#Finding Bootstrap errors from pc.X data
#bootstrap_pcx = stat.bootstrap(pc.X[:,0],N_iter=1000)
#figure(1)
#clf()
#hist(bootstrap_pcx.boot_sample,bins=arange(-0.2,0.2,0.01))

#Use bootstrapping to get a standard deviation for a gaussian that then scatters the data points,
#Using those scattered points we see which class they fall in
#i.e. How robust are these classes?
#N_iter=1000
#boot = stat.gaussian_flucts(pc.X,N_iter=N_iter)
#class_scatter = where(boot.new_labels != boot.old_labels)[0]
#class_scatter_percent = float(len(class_scatter))/len(boot.old_labels)
#print "Using ", N_iter, " iterations, classes are ", class_scatter_percent, "% robust"
