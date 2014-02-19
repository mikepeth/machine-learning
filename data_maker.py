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
import os
import pyfits
from pygoods import *
from scipy import stats

def create_complete_data_mwcat(catalog,zmin=0,zmax=10):
	good_data = where((catalog['Flux_F160W_hst'] > 0.0) & (catalog['Flux_f125W_hst']> 0.0) & \
			  (catalog['Flux_f814W_hst']> 0.0) & (catalog['Flux_f606W_hst'] > 0.0)& \
			  (catalog['Flux_Ks_hawki'] > 0.0) & (catalog['Flux_Y_hawki'] > 0.0)& \
			  (catalog['Flux_U_cfht'] > 0.0) & (catalog['Flux_B_subaru'] > 0.0)& \
			  (catalog['Flux_V_subaru'] > 0.0) & (catalog['Flux_R_subaru'] > 0.0)& \
			  (catalog['Flux_i_subaru'] > 0.0) & (catalog['Flux_z_subaru'] > 0.0) & \
			  (catalog['Photo_z'] > zmin) & (catalog['Photo_z']  < zmax) & \
			  (catalog['Flux_J_ukidss_DR8'] > 0.0) & (catalog['Flux_H_ukidss_DR8'] > 0.0) & \
			  (catalog['Flux_K_ukidss_DR8'] > 0.0) & (catalog['Flux_ch1_seds'] > 0.0) & \
			  (catalog['Flux_ch2_seds'] > 0.0) & (catalog['Flux_ch3_spuds'] > 0.0) & \
			  (catalog['Flux_ch4_spuds'] > 0.0) & (catalog['Flux_z_subaru'] > 0.0))[0]
	return good_data

def create_complete_data_morphcat(catalog,zmin=0,zmax=10,band='J'):
	good_data = where((catalog['SExMAG_%s' % (band)] > 0.0) &  (catalog['SN_%s' % (band)] >= 4.0) & \
			  (catalog['Photo_z'] > zmin) &  (catalog['Photo_z'] < zmax)  &  (catalog['FLAG_%s' % (band)] == 0) & (catalog['C_%s' % (band)] > 0) & \
			  (catalog['log_Stellar_mass'] > 10.0))[0]		
	return good_data

def select_morph_matrix(morph_cat,morph_cat_index,fields,use_flux=False):
	#Iterate over mw_cat._colnames, photz_cat._colnames, morph_cat.names and concatenate these into a single data matrix
	#Eliminate SExID, RA, DEC from morph_cat
	#Eliminate id, ra, dec, f160w_limiting_magnitude,flag,class_star,fluxerr*,spec-z,reference from mw_cat
	#Eliminate id from photoz_mass_cat
	#data_names = morph_cat.names+mw_cat._colnames+photz_cat._colnames
	data = []
	data_names = []
	
	if type(morph_cat) is pyfits.fitsrec.FITS_rec:
		# grabs column data from a FITS file
		if len(fields) > 0:
			for field in fields:
				if field in morph_cat.names and 'R_PET' not in field:
					data = data+[morph_cat[field][morph_cat_index]]
					data_names = data_names+[field]
					#if 'R_PET' in field:
					#	cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' : 0.72}
					#	cosmo = cd.set_omega_k_0(cosmo) #establishes lambda-CDM
					#	ang_size = morph_cat[field][morph_cat_index] #angular size in arcseconds
					#	zp = photz_cat.photo_z[mw_cat_index] #photometric redshifts
					#	d_a = cd.angular_diameter_distance(zp, **cosmo) #distance in Mpc
					#	physical_size = ang_size*4.84813681e-6*d_a*1000. #convert arcsec to radians and Mpc into kpc
					#	data = data+[physical_size]
					#	data_names = data_names+['R_kpc']

	if use_flux == True:
		for par in morph_cat.names:
			if 'Flux' in par and 'Fluxerr' not in par and 'pet' not in par:
				data = data+[morph_cat[par][morph_cat_index]]
				data_names = data_names+[par]
	
	combined_data_string = array(data)
	combined_data = zeros(shape(combined_data_string))
	for i in range(len(data)):
			for j in range(len(data[0])):
				combined_data[i][j] = float(combined_data_string[i][j])
			
	return combined_data,data_names
