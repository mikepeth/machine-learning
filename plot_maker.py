#!/usr/bin/env python

"""
This script will combine the UDS Multi-wavelength catalog, Photo-z/Stellar Mass catalog and Gini-M20 morphology catalog into a single catalog that
will then be run with PCA to determine the important dimensions

UDS.Multi.121221_2ndrelease.flux.cat  - Multiwavelength Catalog (fluxes)
UDS.PhotozMasses_121221.cat - Photo-z/Stellar mass catalog
uds_2epoch_wfc3_f125w_060mas_v0.3_galfit.cat - J-band Galfit
uds_2epoch_wfc3_f160w_060mas_v0.3_galfit.cat - H band Galfit
udse2_mo_h_110613_a_gm20.fits - CAS/GM20 Morphology (not based on Multi-wavelength catalog, so need to catalog match, IDs won't align)

"""

from numpy import *
from scipy import stats
import os
import matplotlib.pyplot as plt
import matplotlib
#import scipy
import rgb_server as rs
import random
from astropy import cosmology
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, \
     AnnotationBbox
import scipy.stats.distributions as dist #Binomial Errorbars, Cameron 2011

import stat_tools as stat



def pca_plots(pc_x,figname):
	plt.figure(figsize=(20,20))
	plt.clf()
	#rc('text', usetex=True)
	dim = 4
	subfig=1
	for col in range(dim+1):
		for row in range(dim+1):
			if row > col:
				plt.subplot(dim,dim,subfig)
				plt.hexbin(pc_x[col],pc_x[row],cmap=plt.cm.YlOrRd_r,extent=[-4,4,-4,4],gridsize=50)
				plt.colorbar()
				#plt.plot(pc_x[col],pc_x[row],'.k',ms=1)
				plt.xlabel('PC%s' % (col+1))
				plt.ylabel('PC%s' % (row+1))
				plt.xlim(-4,4)
				plt.ylim(-4,4)
				subfig+=1
		subfig+=col
	
	plt.savefig(figname)
	return

def pca_plots_z(pc_x,redshift_range,passband):
	plt.figure(figsize=(10,10))
	plt.clf()
	dim = 3
	pcx = 0
	pcy = 1
	
	for zz in range(len(redshift_range)-1):
		plt.subplot(dim,dim,zz)
		plt.hexbin(pc_x[zz].T[pcx],pc_x[zz].T[pcy],cmap=plt.cm.YlOrRd_r,extent=[-4,4,-4,4],gridsize=50)#,'.k',ms=1)
		plt.colorbar()
		if zz == 0 or zz==4:
			plt.xlabel('PC%s'%(pcx+1))
			plt.ylabel('PC%s'%(pcy+1))
		if zz == 5:
			plt.xlabel('PC%s'%(pcx+1))
		if zz==1:
			plt.ylabel('PC%s'%(pcy+1))
			plt.text(-3.5,-3.5,passband,color='white')
		plt.text(-3.5,3.5,'%s < z < %s' % (redshift_range[zz],redshift_range[zz+1]),color='white')
		
		plt.xlim(-4,4)
		plt.ylim(-4,4)
	return

def pca_plots_z_morph(pc_x,redshift_range,passband,rf,filename):
	plt.figure(figsize=(10,10))
	plt.clf()

	dim = 3
	num=0
	
	for pcx in range(dim):
		for pcy in range(dim):
			if pcy > pcx:	
				for zz in range(len(passband)):	
					ax=plt.subplot(dim,len(passband),num+1)
					plt.subplots_adjust(wspace=0,hspace=0.2)
					
					plt.plot(pc_x[zz].T[pcx],pc_x[zz].T[pcy],'.k',ms=4)
					
					plt.xlabel('PC%s'%(pcx+1))
					ax.set_xticks((-3,-2,-1,0,1,2,3))
					if zz == 0:
						plt.ylabel('PC%s'%(pcy+1))
						ax.set_yticks((-3,-2,-1,0,1,2,3))
					else:
						ax.set_yticklabels(())
	
					plt.text(-3.5,3.5,'%s < z < %s' % (redshift_range[2*zz],redshift_range[2*zz+1]),color='black')
					plt.text(-3.5,3.0,passband[zz],color='black')
					plt.xlim(-4,4)
					plt.ylim(-4,4)
						
					num+=1	
	plt.suptitle('Rest-frame SDSS %s Morphology'% (rf))
	plt.savefig(filename)
	return

def pca_plots_z_morph_viztype(catalog,good_data,pc_x,redshift_range,passband,rf,viztype,color_viz,filename):
	plt.figure(figsize=(10,10))
	plt.clf()

	dim = 3
	num=0
	
	for pcx in range(dim):
		for pcy in range(dim):
			if pcy > pcx:	
				for zz in range(len(passband)):	
					ax=plt.subplot(dim,len(passband),num+1)
					plt.subplots_adjust(wspace=0,hspace=0.2)
					pca = pc_x[zz].T[pcx]
					pcb = pc_x[zz].T[pcy]

					if viztype == 'merger' or viztype == 'irr':
						viztype_idx = where(catalog[viztype][good_data[zz]] > 0.6)[0]
					if viztype == 'disk':
						viztype_idx = where((catalog[viztype][good_data[zz]] > 0.6) & (catalog['spheroid'][good_data[zz]] < 0.4))[0]
					if viztype == 'spheroid':
						viztype_idx = where((catalog[viztype][good_data[zz]] > 0.6) & (catalog['disk'][good_data[zz]] < 0.4))[0]

					#print len(viztype_idx)

					if len(viztype_idx) > 1:
						plt.plot(pca[viztype_idx],pcb[viztype_idx],marker='.',color=color_viz,ms=12,linestyle='None')
					if len(viztype_idx) < 1:
						plt.plot(pca,pcb,'.w',ms=1)
					
					plt.xlabel('PC%s'%(pcx+1))
					ax.set_xticks((-3,-2,-1,0,1,2,3))
					if zz == 0:
						plt.ylabel('PC%s'%(pcy+1))
						ax.set_yticks((-3,-2,-1,0,1,2,3))
					else:
						ax.set_yticklabels(())
	
					plt.text(-3.5,3.5,'%s < z < %s' % (redshift_range[2*zz],redshift_range[2*zz+1]),color='black')
					plt.text(-3.5,3.0,passband[zz],color='black')
					plt.text(-3.5,2.5,viztype,color='black')
					plt.xlim(-4,4)
					plt.ylim(-4,4)
						
					num+=1	
	plt.suptitle('Rest-frame SDSS %s Morphology'% (rf))
	plt.savefig(filename)
	return

def pca_plots_z_morph_scatter_multi(catalog,good_data,pc_x,redshift_range,passband,rf,filename,parameter_scatter='SFR',vmin_scatter=0,vmax_scatter=10):
	plt.figure(figsize=(20,20))
	plt.clf()
	cm = matplotlib.cm.get_cmap('YlGn')
	color_bins = 4
	cm_list = [cm(i) for i in range(cm.N)]

	dim = 3
	num=0
	
	for pcx in range(dim):
		for pcy in range(dim):
			if pcy > pcx:	
				for zz in range(len(passband)):	
					ax=plt.subplot(dim,len(passband),num+1)
					plt.subplots_adjust(wspace=0,hspace=0.2)
					plt.scatter(pc_x[zz].T[pcx],pc_x[zz].T[pcy],c=catalog[parameter_scatter][good_data[zz]],cmap=cm,s=20,vmin=vmin_scatter,vmax=vmax_scatter,alpha=0.5)
					     					
					plt.xlabel('PC%s'%(pcx+1))
					ax.set_xticks((-3,-2,-1,0,1,2,3))
					if zz == 0:
						plt.ylabel('PC%s'%(pcy+1))
						ax.set_yticks((-3,-2,-1,0,1,2,3))
					else:
						ax.set_yticklabels(())
	
					plt.text(-3.5,3.5,'%s < z < %s' % (redshift_range[2*zz],redshift_range[2*zz+1]),color='black')
					plt.text(-3.5,3.0,passband[zz],color='black')
					plt.xlim(-4,4)
					plt.ylim(-4,4)

					for cb in range(color_bins):
						#print cb
						binwidth = vmax_scatter - vmin_scatter
						#print binwidth
						#print vmin_scatter+(binwidth*cb/color_bins)
						#print vmin_scatter+(binwidth*(cb+1)/color_bins)
						
						bin_low = vmin_scatter+(binwidth*cb/color_bins)
						if cb == 0:
							bin_low = -99
						bin_high = vmin_scatter+(binwidth*(cb+1)/color_bins)
						if cb == max(range(color_bins)):
							bin_high = 99
							
						binned_data = where((catalog[parameter_scatter][good_data[zz]] >= bin_low) &  \
								    (catalog[parameter_scatter][good_data[zz]] < bin_high))[0]
						#print len(binned_data)
						if len(binned_data) > 1:
							plt.plot(mean(pc_x[zz].T[pcx][binned_data]),mean(pc_x[zz].T[pcy][binned_data]), \
								 color=cm_list[int(256*((2*cb+1)/(color_bins*2.0)))],marker='.',ms=30.0,alpha=0.75,markeredgecolor='black')

					num+=1
						
	plt.suptitle('Rest-frame SDSS %s Morphology by %s'% (rf,parameter_scatter))
	plt.colorbar()

	plt.savefig(filename)
	return


def plotIsomap(pc_x,figname):
	plt.figure(figsize=(15,15))
	plt.clf()
	#rc('text', usetex=True)
	dim = 4
	subfig=1
	x_min = -20.0
	x_max = 20.0
	y_min = -20.0
	y_max = 20.0
	orange=[[x_min,x_max],[y_min,y_max]]
	n_bins=50
	for col in range(dim+1):
		for row in range(dim+1):
			if row > col:
				plt.subplot(dim,dim,subfig)
				plt.plot(pc_x[:,col],pc_x[:,row],'.k',ms=1)
				#hist2d,xe,ye= histogram2d(pc_x[:,col],pc_x[:,row], range=orange ,bins=n_bins)
				#x = linspace(x_min,y_max,n_bins)
				#y = linspace(y_min,y_max,n_bins)
				#extent = [xe[0],xe[-1],ye[0],ye[-1]]
				#xm, ym = griddata((xe,ye))

				plt.xlabel('IM%s' % (col+1))
				plt.ylabel('IM%s' % (row+1))
				plt.xlim(x_min,x_max)
				plt.ylim(y_min,y_max)
				subfig+=1
		subfig+=col

	plt.savefig(figname)
	return

def plotIsomap_z(pc_x,redshift_range,pcx,pcy,filename):
	plt.figure(figsize=(10,10))
	plt.clf()
	dim = 3
#	pcx  = 3
#	pcy = 4
	square = 4 #X,Y +,- limits
	
	for zz in range(len(redshift_range)-1):
		plt.subplot(dim,dim,zz)
		plt.hexbin(pc_x[zz].T[pcx],pc_x[zz].T[pcy],cmap=plt.cm.YlOrRd_r,extent=[-1*square,square,-1*square,square],gridsize=50)#,'.k',ms=1)
		#plt.plot(pc_x[zz].T[pcx],pc_x[zz].T[pcy],'.k',ms=1)
		#plt.colorbar()
		if zz == 0 or zz==4:
			plt.xlabel('IM%s'%(pcx+1))
			plt.ylabel('IM%s'%(pcy+1))
		if zz == 5:
			plt.xlabel('IM%s'%(pcx+1))
		if zz==1:
			plt.ylabel('IM%s'%(pcy+1))
		plt.title('%s < z < %s' % (redshift_range[zz],redshift_range[zz+1]))
		plt.xlim(-1*square,square)
		plt.ylim(-1*square,square)
	plt.savefig(filename)



def pca_scree(eigenvalues,filename):
       	plt.figure()
	plt.clf()
	plt.plot(arange(len(eigenvalues))+1,eigenvalues,'-k')
	plt.plot(arange(len(eigenvalues))+1,eigenvalues,'.k',ms=6)
	plt.xlim(0,7)
	plt.ylim(0.0,0.50)
	plt.xlabel('PC #')
	plt.ylabel('Proportion of Variance along PC')
	plt.savefig(filename)
	return

def pca_var_plots_z(eigenvalues,redshift_range,passband,filename):
	plt.figure(figsize=(10,10))
	plt.clf()
	dim = 3
	
	for zz in range(len(redshift_range)-1):
		plt.subplot(dim,dim,zz)
		plt.plot(arange(len(eigenvalues[zz]))+1,eigenvalues[zz],'.k')
		plt.xlim(-1,10)
		plt.ylim(0.0,1.0)
		plt.xlabel('PC #')
		plt.ylabel('Proportion of Variance along PC')
		plt.text(4,0.7,'%s < z < %s' % (redshift_range[zz],redshift_range[zz+1]),color='black')
	plt.savefig(filename)
	return

def pca_var_plots_z_morph(eigenvalues,redshift_range,passbands,rf,filename):
	plt.figure(figsize=(10,10))
	plt.clf()
	dim = 2
	
	for zz in range(len(passbands)):
		plt.subplot(dim,dim,zz+1)
		plt.plot(arange(len(eigenvalues[zz]))+1,eigenvalues[zz],'.k')
		plt.xlim(0,7)
		plt.ylim(0.0,0.45)
		plt.xlabel('PC #')
		plt.ylabel('Proportion of Variance along PC')
		plt.text(4,0.35,'%s < z < %s' % (redshift_range[2*zz],redshift_range[2*zz+1]),color='black')
		plt.text(4,0.3,passbands[zz],color='black')
	plt.suptitle('Rest-frame SDSS %s Morphology' %(rf))
	plt.savefig(filename)
	return

def pca_plots_z_morph_hexbin(pc_x,redshift_range,passband,rf,filename):

	plt.figure(figsize=(15,10))
	plt.clf()
	dim = 3
	num=0
	
	for pcx in range(dim):
		for pcy in range(dim):
			if pcy > pcx:	
				for zz in range(len(passband)):
					plt.subplot(dim,len(passband),num+1)
					plt.hexbin(pc_x[zz].T[pcx],pc_x[zz].T[pcy],cmap=plt.cm.YlOrRd_r,extent=[-4,4,-4,4],gridsize=50)
					#plt.plot(pc_x[zz].T[pcx],pc_x[zz].T[pcy],'.k',ms=6)
					plt.xlabel('PC%s'%(pcx+1))
					plt.ylabel('PC%s'%(pcy+1))
					plt.colorbar()
					plt.text(-3.5,3.0,'%s < z < %s' % (redshift_range[2*zz],redshift_range[2*zz+1]),color='white',fontsize=8)
					plt.text(-3.5,2.5,passband[zz],color='white',fontsize=8)
					#plt.text(-3.5,3.0,'%s < z < %s' % (redshift_range[2*zz],redshift_range[2*zz+1]),color='black',fontsize=8)
					#plt.text(-3.5,2.5,passband[zz],color='black',fontsize=8)
					plt.xlim(-4,4)
					plt.ylim(-4,4)
					num +=1
						

					#if (zz == 0) & (pcx == 0) & (pcy == 2):
					#	m, b, r, p, std_err = stats.linregress(pc_x[zz].T[pcx],pc_x[zz].T[pcy])
					#	print m, b, r, p
					#	xx = arange(-10,10.,0.1)
					#	yy = xx*m+b
					#	plt.plot(xx,yy,'-k')
						
	plt.suptitle('Rest-frame SDSS %s Morphology'% (rf))
	plt.savefig(filename)
	return


def diffMap_scatter_plots(catalog,good_data,dm,parameter_scatter='SFR',vmin_scatter=0,vmax_scatter=10,filename='dmap.pdf'):
	'''
	Scatter Plot from Diffusion Map output using different variables
	catalog: basis set of non-parametric morphologies
	good_data: indicies of all data that fits our desired sample
	dm: diffusion map coordinates
	parameter_scatter: the parameter to use to color scatter plot; default is star formation rate
	vmin_scatter: minimum value for scatter
	vmax_scatter: maximum value for scatter
	'''
	plt.figure(figsize=(18,6))
	plt.clf()
	cm = matplotlib.cm.get_cmap('YlGn')
	cm_list = [cm(i) for i in range(cm.N)]

	#dim = 3
	#num=0

	'''
	for pcx in range(dim):
		for pcy in range(dim):
			if pcy > pcx:	

				ax=plt.subplot(1,dim,num+1)
				plt.subplots_adjust(wspace=0,hspace=0.2)
	'''
	plt.scatter(dm[:,0],dm[:,1],c=catalog[parameter_scatter][good_data],cmap=cm,s=20,vmin=vmin_scatter,vmax=vmax_scatter,alpha=0.5)

	plt.xlabel(r'$\lambda_1\Psi_1$')
	plt.ylabel(r'$\lambda_2\Psi_2$')
#	plt.xlabel(r'$\lambda_{%s}\Psi_{%s}$' % (pcx+1))
#	plt.ylabel(r'$\lambda_{%s}\Psi_{%s}$' % (pcy+1))
	#ax.set_xticks((-3,-2,-1,0,1,2,3))
	#ax.set_yticks((-3,-2,-1,0,1,2,3))
	#ax.set_yticklabels(())

	#plt.text(-3.5,3.5,'%s < z < %s' % (redshift_range[2*zz],redshift_range[2*zz+1]),color='black')
	#plt.text(-3.5,3.0,passband,color='black')
	plt.title(parameter_scatter)
	#if pcx == 0:
	plt.xlim(-0.02,0.05)
	plt.ylim(-2,1)

				
					
	#			num+=1
						
	plt.colorbar()
	plt.savefig(filename)
	return

def morph_error(data,band='J'):

	mag_range = arange(20,27,0.5)
	if band=='J':
		xtitle = 'F125W Magnitude'
	if band=='H':
		xtitle = 'F160W Magnitude'

	#CONCENTRATION
	plt.close('all')
	plt.figure(figsize=(15,10))
	plt.subplot(3,3,1)
	plt.plot(data.mag,data.c_error,'.k')
	plt.ylim(-3,3)
	plt.xlim(25,18)
	plt.ylabel('C (GOODS-S) - C (UDF) ')
	plt.xlabel(xtitle)

	#Binning for error bars
	std_error = zeros(len(mag_range))
	avg_error = zeros(len(mag_range))
	for i in range(len(mag_range)-1):
		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
					 (data.c_error > -10.0) &  (data.c_error < 10.0) & (isfinite(data.c_error)))[0]
		#print len(brightness_limit)
		std_error[i] = std(data.c_error[brightness_limit])
		avg_error[i] = mean(data.c_error[brightness_limit])
	#print avg_error
	#print mag_range
	plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)
	plt.plot(arange(100),zeros(100),color='blue',linewidth=2)

	
	#GINI COEFFICIENT
	plt.subplot(3,3,2)
	plt.plot(data.mag,data.g_error,'.k')
	plt.ylim(-0.25,0.25)
	plt.xlim(25,18)
	plt.ylabel('G (GOODS-S) - G (UDF) ')
	plt.xlabel(xtitle)

	#Binning for error bars
	std_error = zeros(len(mag_range))
	avg_error = zeros(len(mag_range))
	for i in range(len(mag_range)-1):
		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
					 (data.g_error > -10.0) &  (data.g_error < 10.0) & (isfinite(data.g_error)))[0]
		#print len(brightness_limit)
		std_error[i] = std(data.g_error[brightness_limit])
		avg_error[i] = mean(data.g_error[brightness_limit])
	#print avg_error
	#print mag_range
	plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)
	plt.plot(arange(100),zeros(100),color='blue',linewidth=2)

	#M20 MOMENT OF LIGHT
	plt.subplot(3,3,3)
	plt.plot(data.mag,data.m20_error,'.k')
	plt.ylim(-1,1)
	plt.xlim(25,18)
	plt.ylabel(r'M$_{20}$ (GOODS-S) - M$_{20}$ (UDF) ')
	plt.xlabel(xtitle)

	#Binning for error bars
	std_error = zeros(len(mag_range))
	avg_error = zeros(len(mag_range))
	for i in range(len(mag_range)-1):
		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
					 (data.m20_error > -10.0) &  (data.m20_error < 10.0) & (isfinite(data.m20_error)))[0]
		#print len(brightness_limit)
		std_error[i] = std(data.m20_error[brightness_limit])
		avg_error[i] = mean(data.m20_error[brightness_limit])
	#print avg_error
	#print mag_range
	plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)
	plt.plot(arange(100),zeros(100),color='blue',linewidth=2)

	#ASYMMETRY
	plt.subplot(3,3,4)
	plt.plot(data.mag,data.a_error,'.k')
	plt.ylim(-1,1)
	plt.xlim(25,18)
	plt.ylabel('A (GOODS-S) - A (UDF) ')
	plt.xlabel(xtitle)

	#Binning for error bars
	std_error = zeros(len(mag_range))
	avg_error = zeros(len(mag_range))
	for i in range(len(mag_range)-1):
		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
					 (data.a_error > -10.0) &  (data.a_error < 10.0) & (isfinite(data.a_error)))[0]
		#print len(brightness_limit)
		std_error[i] = std(data.a_error[brightness_limit])
		avg_error[i] = mean(data.a_error[brightness_limit])
	#print avg_error
	#print mag_range
	plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)
	plt.plot(arange(100),zeros(100),color='blue',linewidth=2)

	#MULTIMODE
	plt.subplot(3,3,5)
	plt.plot(data.mag,data.m_error,'.k')
	plt.ylim(-3,10)
	plt.xlim(25,18)
	plt.ylabel('M (GOODS-S) - M (UDF) ')
	plt.xlabel(xtitle)

	#Binning for error bars
	std_error = zeros(len(mag_range))
	avg_error = zeros(len(mag_range))
	for i in range(len(mag_range)-1):
		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
					 (data.m_error > -10.0) &  (data.m_error < 10.0) & (isfinite(data.m_error)))[0]
		#print len(brightness_limit)
		std_error[i] = std(data.m_error[brightness_limit])
		avg_error[i] = mean(data.m_error[brightness_limit])
	#print avg_error
	#print mag_range
	plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)
	plt.plot(arange(100),zeros(100),color='blue',linewidth=2)

	#INTENSITY
	plt.subplot(3,3,6)
	plt.plot(data.mag,data.i_error,'.k')
	plt.ylim(-1,1)
	plt.xlim(25,18)
	plt.ylabel('I (GOODS-S) - I (UDF) ')
	plt.xlabel(xtitle)

	#Binning for error bars
	std_error = zeros(len(mag_range))
	avg_error = zeros(len(mag_range))
	for i in range(len(mag_range)-1):
		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
					 (data.i_error > -10.0) &  (data.i_error < 10.0) & (isfinite(data.i_error)))[0]
		#print len(brightness_limit)
		std_error[i] = std(data.i_error[brightness_limit])
		avg_error[i] = mean(data.i_error[brightness_limit])
	#print avg_error
	#print mag_range
	plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)
	plt.plot(arange(100),zeros(100),color='blue',linewidth=2)

	#DEVIATION
	plt.subplot(3,3,7)
	plt.plot(data.mag,data.d_error,'.k')
	plt.ylim(-1,1)
	plt.xlim(25,18)
	plt.ylabel('D (GOODS-S) - D (UDF) ')
	plt.xlabel(xtitle)

	#Binning for error bars
	std_error = zeros(len(mag_range))
	avg_error = zeros(len(mag_range))
	for i in range(len(mag_range)-1):
		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
					 (data.d_error > -10.0) &  (data.d_error < 10.0) & (isfinite(data.d_error)))[0]
		#print len(brightness_limit)
		std_error[i] = std(data.d_error[brightness_limit])
		avg_error[i] = mean(data.d_error[brightness_limit])
	#print avg_error
	#print mag_range
	plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)
	plt.plot(arange(100),zeros(100),color='blue',linewidth=2)

	#PETROSIAN RADIUS
	plt.subplot(3,3,8)
	plt.plot(data.mag,data.d_error,'.k')
	plt.ylim(-1,1)
	plt.xlim(25,18)
	plt.ylabel(r'R$_{pet}$ (GOODS-S) - R$_{pet}$ (UDF) ')
	plt.xlabel(xtitle)

	#Binning for error bars
	std_error = zeros(len(mag_range))
	avg_error = zeros(len(mag_range))
	for i in range(len(mag_range)-1):
		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
					 (data.rpet_error > -10.0) &  (data.rpet_error < 10.0) & (isfinite(data.rpet_error)))[0]
		#print len(brightness_limit)
		std_error[i] = std(data.rpet_error[brightness_limit])
		avg_error[i] = mean(data.rpet_error[brightness_limit])
	#print avg_error
	#print mag_range
	plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)
	plt.plot(arange(100),zeros(100),color='blue',linewidth=2)
	
	plt.savefig('morphology_udf_errorbars_%s.png' % (band))

def morph_error_bootstrap(data_class,band='J'):

	if band=='J':
		xtitle = 'F125W Magnitude'
	if band=='H':
		xtitle = 'F160W Magnitude'
		
	data_dict = data_class.__dict__
	parameters = data_dict.keys()
	parameters.remove('sn')

	plt.close('all')
	plt.figure(figsize=(15,15))
	for i in range(len(parameters)):
		if parameters[i] != 'mag':
			plt.subplot(3,3,i+1)
			print "Bootstrapping ", parameters[i]
			#if parameters[i] != 'c_error':
			bootstrap_by_mag = stat.bootstrap_by_mag(data_class.mag,data_dict[parameters[i]])
			#else:
				#print "Selecting only C>0"
				#c_non_negative = where(data_class.c_error > 0.0)[0]
				#bootstrap_by_mag = stat.bootstrap_by_mag(data_class.mag[c_non_negative],data_class.c_error[c_non_negative])
				
			plt.errorbar(bootstrap_by_mag.mag_range+0.25,bootstrap_by_mag.boot_avg,yerr=bootstrap_by_mag.error,color='red',fmt='o')
			plt.xlim(25,19)
			if parameters[i] == 'm_error':
				plt.ylim(-0.2,2)
			else:
				plt.ylim(-0.2,0.2)
			
			plt.ylabel(r'Average $\Delta$ %s' % (parameters[i]))
			plt.xlabel(xtitle)
			plt.plot(arange(0,100),zeros(100),'--k') # Dashed line at zero
	plt.savefig('morph_boot.png')
	return

def latex_table(data,errorbars):
	f = open('pca_latex_medians.txt','w')
	for i in range(len(data[0])):
		for j in range(len(data[0])):
			f.write('%.3f $\pm$ %.3f ' % (data[i][j],errorbars[i][j]))
			if j == len(data[0]) - 1:
				bs = '\\'.__repr__()
				f.write('%s ' % bs)
				f.write('\n')
			else:
				f.write('& ')
	return

def morph_error_bootstrap_percent(catalog,data_class,band='J'):

	magnitude_name = 'SExMAG'+'_'+band+'_UDF'
	udf = where(isfinite(catalog['RA_UDF']) & (catalog['SExMAG_'+band] > 0) & (catalog[magnitude_name] > 0))[0]
	variables = ['A','I','M','G','SExMAG','D','M20','C','Rpet_ell']
	for v in range(len(variables)):
		       variables[v] = variables[v]+'_'+band+'_UDF'

	if band=='J':
		xtitle = 'F125W Magnitude'
	if band=='H':
		xtitle = 'F160W Magnitude'
		
	data_dict = data_class.__dict__
	parameters = data_dict.keys()
	parameters.remove('sn')

	plt.close('all')
	plt.figure(figsize=(15,15))
	for i in range(len(parameters)):
	#	if parameters[i] != 'mag':
		plt.subplot(3,3,i+1)
		print "Bootstrapping ", parameters[i]

		bootstrap_data_by_mag = stat.bootstrap_by_mag(catalog[magnitude_name][udf],catalog[variables[i]][udf])
		bootstrap_by_mag = stat.bootstrap_by_mag(data_class.mag,data_dict[parameters[i]])


		plt.errorbar(bootstrap_by_mag.mag_range+0.25,bootstrap_by_mag.boot_avg/bootstrap_data_by_mag.boot_avg,yerr=bootstrap_by_mag.error,color='red',fmt='o')
		plt.xlim(25,19)
				#if parameters[i] == 'm_error':
				#	plt.ylim(-0.2,2)
				#else:
				#	plt.ylim(-0.2,0.2)

		plt.ylabel('%s (UDF) Percent Difference' % (parameters[i]))
		plt.xlabel(xtitle)
		plt.plot(arange(0,100),zeros(100),'--k') # Dashed line at zero
	plt.savefig('morph_boot_percent_%s.png' % (band))
	return

def morph_error_new(catalog,data,band='J'):

	mag_range = arange(20,26,0.5)
	if band=='J':
		xtitle = 'F125W Magnitude'
	if band=='H':
		xtitle = 'F160W Magnitude'

	variables = ['A','I','M','G','SExMAG','D','M20','C','Rpet_ell']
	for v in range(len(variables)):
		       variables[v] = variables[v]+'_'+band+'_UDF'

	data_dict = data.__dict__
	parameters = data_dict.keys()
	parameters.remove('sn')
	
	plt.close('all')
	plt.figure(figsize=(15,10))
	
	nfig=0
	#Binning for error bars
	for param in parameters:
		std_error = zeros(len(mag_range))
		avg_error = zeros(len(mag_range))
		std_udf = zeros(len(mag_range))
		avg_udf = zeros(len(mag_range))

		magnitude_name = 'SExMAG'+'_'+band+'_UDF'
		udf = where(isfinite(catalog['RA_UDF']) & (catalog['SExMAG_'+band] > 0) & (catalog[magnitude_name] > 0))[0]

		#print
		#print param
		#print
		for i in range(len(mag_range)-1):
			## if param == 'm_error':
			## 	if mag_range[i] > 23:
			## 			brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
			## 			 (data_dict[param] > -20) &  (data_dict[param] < 20) & (isfinite(data_dict[param])))[0]
			## 	else:
			## 		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
			## 			 (data_dict[param] > -10) &  (data_dict[param] < 10) & (isfinite(data_dict[param])))[0]
			
			## elif param == 'c_error':
			## 	brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
			## 			 (data_dict[param] > -2) &  (data_dict[param] < 2) & (isfinite(data_dict[param])))[0]
			## elif param == 'm20_error':
			## 	if mag_range[i] == 20.5:
			## 		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
			## 			 (data_dict[param] > -1) &  (data_dict[param] < 1) & (isfinite(data_dict[param])))[0]
			## 	else:
			## 		brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
			## 			 (data_dict[param] > -0.75) &  (data_dict[param] < 0.75) & (isfinite(data_dict[param])))[0]
			## else:
			## 	brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
			## 			 (data_dict[param] > -0.5) &  (data_dict[param] < 0.5) & (isfinite(data_dict[param])))[0]
			brightness_limit_tot = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) & (isfinite(data_dict[param])))[0]
			
#			if len(brightness_limit_tot) > 0:
#				print mag_range[i]
#				print len(brightness_limit)/float(len(brightness_limit_tot))*100
			
			

			if param == 'mag':
				avg_error[i] = median((data.mag[brightness_limit_tot]-catalog[magnitude_name][udf][brightness_limit_tot])/catalog[magnitude_name][udf][brightness_limit_tot])
				std_error[i] = stat.MAD((data.mag[brightness_limit_tot]-catalog[magnitude_name][udf][brightness_limit_tot])/catalog[magnitude_name][udf][brightness_limit_tot])
				print avg_udf[i]
			else:
				std_error[i] = stat.MAD(data_dict[param][brightness_limit_tot])
				avg_error[i] = median(data_dict[param][brightness_limit_tot])

		
		plt.subplot(3,3,nfig)
		plt.plot(arange(100),zeros(100),color='blue',linewidth=2)

		if param == 'mag':
			plt.plot(data.mag,(data.mag-catalog[magnitude_name][udf])/catalog[magnitude_name][udf],'.k')
			plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)
		else:
			plt.plot(data.mag,data_dict[param],'.k')
			plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)
		
		#plt.plot(data.mag,data_dict[param],'.k')
		#plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)

		if 'm_' in param:
			plt.ylim(-2,10)
		elif 'i_' in param:
			plt.ylim(-0.2,0.6)
		elif param == 'c_error':
			plt.ylim(-1,1)

		elif param == 'mag':
			plt.ylim(-0.005,0.005)
		else:
			plt.ylim(-0.5,0.5)
		plt.xlim(25,18)
		plt.ylabel(r'$\Delta$ %s' % (param))
		plt.xlabel(xtitle)
		nfig+=1
	plt.savefig('morphology_udf_errorbars_%s_v2.png' % (band))


def morph_error_percent_new(catalog,data,band='J'):

	mag_range = arange(20,26,0.5)
	if band=='J':
		xtitle = 'F125W Magnitude'
	if band=='H':
		xtitle = 'F160W Magnitude'

	variables = ['A','I','M','G','SExMAG','D','M20','C','Rpet_ell']
	for v in range(len(variables)):
		       variables[v] = variables[v]+'_'+band+'_UDF'

	data_dict = data.__dict__
	parameters = data_dict.keys()
	parameters.remove('sn')
	
	plt.close('all')
	plt.figure(figsize=(15,10))
	
	nfig=0
	#Binning for error bars
	for param in parameters:
		std_error = zeros(len(mag_range))
		avg_error = zeros(len(mag_range))
		std_udf = zeros(len(mag_range))
		avg_udf = zeros(len(mag_range))

		magnitude_name = 'SExMAG'+'_'+band+'_UDF'
		udf = where(isfinite(catalog['RA_UDF']) & (catalog['SExMAG_'+band] > 0) & (catalog[magnitude_name] > 0))[0]

		print
		print param
		print
		
		
		for i in range(len(mag_range)-1):
			if param == 'm_error':
				brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
							 (data_dict[param]/catalog[variables[nfig]][udf] > -2) &  (data_dict[param]/catalog[variables[nfig]][udf] < 400) \
							 & (isfinite(data_dict[param])) & (catalog[variables[nfig]][udf] > 0))[0]
			elif param == 'mag':
				brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]))[0]
			elif param == 'c_error':
				brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
						 (data_dict[param]/catalog[variables[nfig]][udf] > -1) &  (data_dict[param]/catalog[variables[nfig]][udf] < 1) & \
							 (isfinite(data_dict[param])) & (catalog[variables[nfig]][udf] > 0))[0]
			elif param == 'i_error':
				brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
						 (data_dict[param]/catalog[variables[nfig]][udf] > -2) &  (data_dict[param]/catalog[variables[nfig]][udf] < 30) & \
							 (isfinite(data_dict[param])) & (catalog[variables[nfig]][udf] > 0))[0]
				#print len(brightness_limit)
			elif param == 'd_error':
				brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
						 (data_dict[param]/catalog[variables[nfig]][udf] > -1) &  (data_dict[param]/catalog[variables[nfig]][udf] < 10) & \
							 (isfinite(data_dict[param])) & (catalog[variables[nfig]][udf] > 0))[0]

			elif param == 'a_error':
				brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
						 (data_dict[param]/catalog[variables[nfig]][udf] > -10) &  (data_dict[param]/catalog[variables[nfig]][udf] < 10) & \
							 (isfinite(data_dict[param])) & (catalog[variables[nfig]][udf] > 0))[0]
			elif param == 'm20_error':
				brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
							 (data_dict[param]/catalog[variables[nfig]][udf] > -0.75) &  (data_dict[param]/catalog[variables[nfig]][udf] < 0.75) & \
							 (isfinite(data_dict[param])) & (catalog[variables[nfig]][udf] > -5))[0]
			else:
				brightness_limit = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) &
						 (data_dict[param]/catalog[variables[nfig]][udf] > -0.5) &  (data_dict[param]/catalog[variables[nfig]][udf] < 0.5) & \
							 (isfinite(data_dict[param])) & (catalog[variables[nfig]][udf] > 0))[0]
			brightness_limit_tot = where((data.mag > mag_range[i]) & (data.mag <= mag_range[i+1]) & (isfinite(data_dict[param])) &
						     ((catalog[variables[nfig]][udf] > 0)))[0]

			
			std_error[i] = stat.MAD(data_dict[param][brightness_limit])
			avg_error[i] = mean(data_dict[param][brightness_limit])

			

			if param == 'mag':
				avg_udf[i] = median((data.mag[brightness_limit]-catalog[magnitude_name][udf][brightness_limit])/catalog[magnitude_name][udf][brightness_limit])
				std_udf[i] = stat.MAD((data.mag[brightness_limit]-catalog[magnitude_name][udf][brightness_limit])/catalog[magnitude_name][udf][brightness_limit])
				print avg_udf[i]
			else:
				avg_udf[i] = median(data_dict[param][brightness_limit]/catalog[variables[nfig]][udf][brightness_limit])
				std_udf[i] = stat.MAD(data_dict[param][brightness_limit]/catalog[variables[nfig]][udf][brightness_limit])

			#if len(brightness_limit_tot) > 0:
			#	print mag_range[i]
			#	print len(brightness_limit)/float(len(brightness_limit_tot))*100

		
		#print std_udf
		plt.subplot(3,3,nfig)
		plt.plot(arange(100),zeros(100),color='blue',linewidth=2)
	
		if param == 'mag':
			plt.plot(data.mag,(data.mag-catalog[magnitude_name][udf])/catalog[magnitude_name][udf],'.k')
			plt.errorbar(mag_range+0.25,avg_udf,yerr=std_udf,color='r',linewidth=3)
		else:
			plt.plot(data.mag,data_dict[param]/catalog[variables[nfig]][udf],'.k')
			plt.errorbar(mag_range+0.25,avg_udf,yerr=std_udf,color='r',linewidth=3)
		
                #plt.plot(data.mag,data_dict[param],'.k')
		#plt.errorbar(mag_range+0.25,avg_error,yerr=std_error,color='r',linewidth=3)

		percent_dif = True #Says Im finding (GOODSS-UDF)/UDF
		if 'm_' in param:
			plt.ylim(-10,300)
		elif 'i_' in param:
			plt.ylim(-5,10)
		elif param == 'c_error':
			plt.ylim(-0.5,0.5)
		elif param == 'd_error' and percent_dif == True:
			plt.ylim(-3,10)
		elif param == 'a_error' and percent_dif == True:
			plt.ylim(-5,5)
		elif param == 'mag':
			plt.ylim(-0.005,0.005)
		else:
			plt.ylim(-0.4,0.4)
		plt.xlim(25,18)
		#plt.ylabel(r'$\Delta$ %s' % (param))
		plt.ylabel(r'UDF Percent %s' % (param))
		plt.xlabel(xtitle)
		nfig+=1
	plt.savefig('morphology_udf_errorbars_percent_%s.png' % (band))

def vis_class(catalog,classification):
	disk_tot = where((catalog['disk'] > 2/3.) & (catalog['spheroid'] < 2/3.))[0]
	spheroid_tot = where((catalog['spheroid'] > 2/3.) & (catalog['disk'] < 2/3.))[0]
	irregular_tot =  where((catalog['irr'] > 2/3.) & (catalog['disk'] < 2/3.) & (catalog['spheroid'] < 2/3.))[0]
	disk_sph_tot =  where((catalog['disk'] > 2/3.) & (catalog['spheroid'] > 2/3.))[0]

	#classification = classification[0:len(catalog)] #Works for UDS catalog only, hese galaxies comprise the first block of data

	#p_lower = dist.beta.ppf((1-c)/2.,k+1,n-k+1)
	#p_upper = dist.beta.ppf(1-(1-c)/2.,k+1,n-k+1)
	#c = confidence interval
	#k = success rate
	#n = sample size

	c=0.683 #Confidence Interval, 1-sigma
	
	ndisk_class = []
	ndisk_class_upper = []
	ndisk_class_lower = []
	
	nsphd_class = []
	nirrg_class = []
	ndsph_class = []

	ndisk_class_ofclass = []
	nsphd_class_ofclass = []
	nirrg_class_ofclass = []
	ndsph_class_ofclass = []

	ndisk_class_ofclass_upper = []
	nsphd_class_ofclass_upper = []
	nirrg_class_ofclass_upper = []
	ndsph_class_ofclass_upper = []
	ndisk_class_ofclass_lower = []
	nsphd_class_ofclass_lower = []
	nirrg_class_ofclass_lower = []
	ndsph_class_ofclass_lower = []

	
	for n in unique(classification):
		group = where(classification == n)[0]
		disks_class = float(len(where((catalog['disk'][group] > 2/3.) & (catalog['spheroid'][group]  < 2/3.))[0]))
		spheroid_class = float(len(where((catalog['spheroid'][group]  > 2/3.) & (catalog['disk'][group]  < 2/3.))[0]))
		irregular_class =  float(len(where((catalog['irr'][group]  > 2/3.) & (catalog['disk'][group]  < 2/3.) & (catalog['spheroid'][group] < 2/3.))[0]))
		disk_sph_class =  float(len(where((catalog['disk'][group]  > 2/3.) & (catalog['spheroid'][group]  > 2/3.))[0]))

		#Morphology as percentage of total classifications
		#ndisk_class = ndisk_class + [100.0*(disks_class)/disk_tot]
		#ndisk_class_upper = ndisk_class_upper + [100*dist.beta.ppf(1-(1-c)/2.,disks_class+1,disk_tot-disks_class+1)]
		#ndisk_class_lower = ndisk_class_lower + [100*dist.beta.ppf((1-c)/2.,disks_class+1,disk_tot-disks_class+1)]
		#nsphd_class = nsphd_class + [100.0*(spheroid_class)/spheroid_tot]
		#nirrg_class = nirrg_class + [100.0*(irregular_class)/irregular_tot]
		#ndsph_class = ndsph_class + [100.0*(disk_sph_class)/disk_sph_tot]

		#Morphology as percentage of Group 
		ngal = float(len(where(catalog['disk'][group] >= 0.0)[0]))
		ndisk_class_ofclass = ndisk_class_ofclass + [100.0*(disks_class)/ngal]
		#ndisk_class_ofclass_upper = ndisk_class_ofclass_upper + [100*dist.beta.ppf(1-(1-c)/2.,disks_class+1,ngal-disks_class+1)]
		#ndisk_class_ofclass_lower = ndisk_class_ofclass_lower + [100*dist.beta.ppf((1-c)/2.,disks_class+1,ngal-disks_class+1)]
		
		nsphd_class_ofclass = nsphd_class_ofclass + [100.0*(spheroid_class)/ngal]
		#nsphd_class_ofclass_upper = nsphd_class_ofclass_upper + [100*dist.beta.ppf(1-(1-c)/2.,spheroid_class+1,ngal-spheroid_class+1)]
		#nsphd_class_ofclass_lower = nsphd_class_ofclass_lower + [100*dist.beta.ppf((1-c)/2.,spheroid_class+1,ngal-spheroid_class+1)]
		
		nirrg_class_ofclass = nirrg_class_ofclass + [100.0*(irregular_class)/ngal]
		#nirrg_class_ofclass_upper = nirrg_class_ofclass_upper + [100*dist.beta.ppf(1-(1-c)/2.,irregular_class+1,ngal-irregular_class+1)]
		#nirrg_class_ofclass_lower = nirrg_class_ofclass_lower + [100*dist.beta.ppf((1-c)/2.,irregular_class+1,ngal-irregular_class+1)]
		
		ndsph_class_ofclass = ndsph_class_ofclass + [100.0*(disk_sph_class)/ngal]
		#ndsph_class_ofclass_upper = ndsph_class_ofclass_upper + [100*dist.beta.ppf(1-(1-c)/2.,disk_sph_class+1,ngal-disk_sph_class+1)]
		#ndsph_class_ofclass_lower = ndsph_class_ofclass_lower + [100*dist.beta.ppf((1-c)/2.,disk_sph_class+1,ngal-disk_sph_class+1)]

	#plt.figure()
	#plt.plot(unique(classification)-0.1,ndisk_class, '*k', label='Disks',ms=15)
	#plt.errorbar(unique(classification)-0.1,ndisk_class,yerr=[ndisk_class_lower,ndisk_class_upper],linestyle='',color='k')
	
	#plt.plot(unique(classification)-0.2,nsphd_class, 'or', label='Spheroids',ms=15)
	#plt.plot(unique(classification)+0.2,nirrg_class, '.b', label='Irregulars',ms=15)
	#plt.plot(unique(classification)+0.1,ndsph_class, '+g', label='Disk+Sph',ms=15,mew=4)
	#plt.legend(loc=1)
	#plt.xlabel('Group Number')
	#plt.ylabel('Fraction')
	#plt.ylim(-10,100)
	#plt.xticks(unique(classification),unique(classification))
	#plt.title('Percentage of Total Morphology Bin F125W, 1.36 < z < 1.97')
	#plt.savefig('class_demography_tot.pdf')

	plt.figure()
	plt.plot(unique(classification)-0.1,ndisk_class_ofclass, '*k', label='Disks',ms=15)
	#plt.errorbar(unique(classification)-0.1,ndisk_class_ofclass,yerr=[ndisk_class_ofclass_lower,ndisk_class_ofclass_upper],linestyle='',color='k')
	plt.plot(unique(classification)-0.2,nsphd_class_ofclass, 'or', label='Spheroids',ms=15)
	#plt.errorbar(unique(classification)-0.2,nsphd_class_ofclass,yerr=[nsphd_class_ofclass_lower,nsphd_class_ofclass_upper],linestyle='',color='r')
	plt.plot(unique(classification)+0.2,nirrg_class_ofclass, '.b', label='Irregulars',ms=15)
	#plt.errorbar(unique(classification)+0.2,nirrg_class_ofclass,yerr=[nirrg_class_ofclass_lower,nirrg_class_ofclass_upper],linestyle='',color='b')
	plt.plot(unique(classification)+0.1,ndsph_class_ofclass, '+g', label='Disk+Sph',ms=15,mew=4)
	#plt.errorbar(unique(classification)+0.1,ndsph_class_ofclass,yerr=[ndsph_class_ofclass_lower,ndsph_class_ofclass_upper],linestyle='',color='g')
	
	plt.legend(loc=7)
	plt.xlabel('Group Number')
	plt.ylabel('Fraction')
	plt.ylim(-10,100)
	plt.xticks(unique(classification),unique(classification))
	plt.title('Percentage of Group F125W, 1.36 < z < 1.97')
	plt.savefig('class_demography_candels.pdf')
		
		
		
def sersic_class(catalog,classification):
	#classification = classification[0:len(catalog)] 
	nlow_class_ofclass = []
	nmid1_class_ofclass = []
	nmid2_class_ofclass = []
	nhigh_class_ofclass = []

	nlow_class_ofclass_upper = []
	nmid1_class_ofclass_upper = []
	nmid2_class_ofclass_upper = []
	nhigh_class_ofclass_upper = []
	nlow_class_ofclass_lower = []
	nmid1_class_ofclass_lower = []
	nmid2_class_ofclass_lower = []
	nhigh_class_ofclass_lower = []

	c=0.683 #Confidence Interval, 1-sigma
	for n in unique(classification):
		group = where(classification == n)[0]
		low_sersic = float(len(where(catalog['gfit_n_j'][group] < 1)[0]))
		mid1_sersic = float(len(where((catalog['gfit_n_j'][group] >=1) & (catalog['gfit_n_j'][group] < 2.5))[0]))
		mid2_sersic = float(len(where((catalog['gfit_n_j'][group] >=2.5) & (catalog['gfit_n_j'][group] < 4))[0]))
		high_sersic = float(len(where((catalog['gfit_n_j'][group] >=4))[0]))
		
		ngal = float(len(group))
		nlow_class_ofclass  = nlow_class_ofclass +  [100.0*(low_sersic)/ngal]
		nlow_class_ofclass_upper = nlow_class_ofclass_upper + [100*dist.beta.ppf(1-(1-c)/2.,low_sersic+1,ngal-low_sersic+1)]
		nlow_class_ofclass_lower = nlow_class_ofclass_lower + [100*dist.beta.ppf((1-c)/2.,low_sersic+1,ngal-low_sersic+1)]
		
		nmid1_class_ofclass = nmid1_class_ofclass + [100.0*(mid1_sersic)/ngal]
		nmid1_class_ofclass_upper = nmid1_class_ofclass_upper + [100*dist.beta.ppf(1-(1-c)/2.,mid1_sersic+1,ngal-mid1_sersic+1)]
		nmid1_class_ofclass_lower = nmid1_class_ofclass_lower + [100*dist.beta.ppf((1-c)/2.,mid1_sersic+1,ngal-mid1_sersic+1)]
		
		nmid2_class_ofclass = nmid2_class_ofclass + [100.0*(mid2_sersic)/ngal]
		nmid2_class_ofclass_upper = nmid2_class_ofclass_upper + [100*dist.beta.ppf(1-(1-c)/2.,mid2_sersic+1,ngal-mid2_sersic+1)]
		nmid2_class_ofclass_lower = nmid2_class_ofclass_lower + [100*dist.beta.ppf((1-c)/2.,mid2_sersic+1,ngal-mid2_sersic+1)]
		
		nhigh_class_ofclass = nhigh_class_ofclass + [100.0*(high_sersic)/ngal]
		nhigh_class_ofclass_upper = nhigh_class_ofclass_upper + [100*dist.beta.ppf(1-(1-c)/2.,high_sersic+1,ngal-high_sersic+1)]
		nhigh_class_ofclass_lower = nhigh_class_ofclass_lower + [100*dist.beta.ppf((1-c)/2.,high_sersic+1,ngal-high_sersic+1)]

	plt.figure()
	plt.plot(unique(classification)-0.1,nlow_class_ofclass, '*k', label='n < 1',ms=15)
	#plt.errorbar(unique(classification)-0.1,nlow_class_ofclass,yerr=[nlow_class_ofclass_lower,nlow_class_ofclass_upper],linestyle='',color='k')
	plt.plot(unique(classification)-0.2,nmid1_class_ofclass, 'or', label='1 < n < 2.5',ms=15)
	#plt.errorbar(unique(classification)-0.2,nmid1_class_ofclass,yerr=[nmid1_class_ofclass_lower,nmid1_class_ofclass_upper],linestyle='',color='r')
	plt.plot(unique(classification)+0.2,nmid2_class_ofclass, '.b', label='2.5 < n < 4',ms=15)
	#plt.errorbar(unique(classification)+0.2,nmid2_class_ofclass,yerr=[nmid2_class_ofclass_lower,nmid2_class_ofclass_upper],linestyle='',color='b')
	plt.plot(unique(classification)+0.1,nhigh_class_ofclass, '+g', label='n > 4',ms=15,mew=4)
	#plt.errorbar(unique(classification)+0.1,nhigh_class_ofclass,yerr=[nhigh_class_ofclass_lower,nhigh_class_ofclass_upper],linestyle='',color='g')
	plt.legend(loc=7)
	plt.xlabel('Group Number')
	plt.ylabel('Fraction')
	plt.ylim(-10,100)
	plt.xticks(unique(classification),unique(classification))
	plt.title('Percentage of Group F125W, 1.36 < z < 1.97')
	plt.savefig('class_demography_candels_sersic.pdf')
		
def pc_plot_parameter(pc,catalog):

	kpc_per_arcmin = cosmology.funcs.kpc_proper_per_arcmin(catalog['Photo_z'])
	kpc_per_arcsec = kpc_per_arcmin.value/60.0 #kpc/arcsec
	reff_kpc = catalog['gfit_sma_j']*kpc_per_arcsec #some number converting redshift and size
	
	plt.figure()
	for pcomp in range(4):
		plt.subplot(2,2,pcomp+1)
		plt.plot(catalog['gfit_n_j'],pc.X[:,pcomp],'.k')
		plt.xlim(0,8)
		plt.ylim(-8,4)
		plt.xlabel('Sersic Index')
		plt.ylabel('PC%s' % str(pcomp+1))
		plt.savefig('demography_candels_sersic.pdf')

	plt.figure()
	for pcomp in range(4):
		plt.subplot(2,2,pcomp+1)
		plt.plot(catalog['gfit_q_j'],pc.X[:,pcomp],'.k')
		plt.xlim(-0.15,1.15)
		plt.ylim(-8,4)
		plt.xlabel('Q Index (B/A)')
		plt.ylabel('PC%s' % str(pcomp+1))
		plt.savefig('demography_candels_ba.pdf')
	
	plt.figure()
	for pcomp in range(4):
		plt.subplot(2,2,pcomp+1)
		plt.plot(reff_kpc,pc.X[:,pcomp],'.k')
		plt.xlim(0,8)
		plt.ylim(-8,4)
		plt.xlabel(r'R$_{eff}$ (kpc)')
		plt.ylabel('PC%s' % str(pcomp+1))
		plt.savefig('demography_candels_reff.pdf')
