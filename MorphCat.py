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
import machine_learning as ml

def create_complete_data_morphcat(self,catalog,zmin=0,zmax=10,band='J'):
	if 'Photo_z' in catalog.names:
		photz_string = 'Photo_z'
	if 'zbest_candels' in catalog.names:
		photz_string = 'zbest_candels'

	if 'log_Stellar_Mass' in catalog.names:
		mass_string = 'log_Stellar_mass'
	if 'FAST_bc03_lmass' in catalog.names:
		mass_string = 'FAST_bc03_lmass'

	masscut_sample_indicies = where((isfinite(catalog['SExMAG_%s' % (band)])) &  (catalog['SN_%s' % (band)] >= 4.0) & \
			  (catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax)  &  (catalog['FLAG_%s' % (band)] == 0) & (catalog['C_%s' % (band)] > 0) & \
			  (catalog[mass_string] > 10.0))[0]		
	return masscut_sample_indicies

def select_morph_matrix(self,morph_cat,fields,use_flux=False):
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
					data = data+[morph_cat[field]]
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
				data = data+[morph_cat[par]]
				data_names = data_names+[par]

	combined_data_string = array(data)
	combined_data = zeros(shape(combined_data_string))
	for i in range(len(data)):
			for j in range(len(data[0])):
				combined_data[i][j] = float(combined_data_string[i][j])

	return combined_data,data_names

class morph_catalog:
	def __init__(self,A,catalog1_sample, A_catalog2=None,A_catalog3=None, catalog2_sample=None,catalog3_sample=None):

		mass_1cat = catalog1_sample['log_stellar_mass']
		sfr_1cat  = catalog1_sample['SFR']
		ra_1cat   = catalog1_sample['RA']
		dec_1cat  = catalog1_sample['DEC']
		photz_1cat = catalog1_sample['Photo_z']
		id_1cat    = catalog1_sample['ID']
		catalog_1cat = zeros(len(A)) #0 = UDS, 1 = COSMOS, 2= GOODS-S


		if A_catalog2 != None:
			A_2catalogs = concatenate([A,A_catalog2],axis=1)
			mass_2cat = concatenate([catalog1_sample['log_stellar_mass'],catalog2_sample['FAST_bc03_lmass']])
			sfr_2cat  = concatenate([catalog1_sample['SFR'],catalog2_sample['SFR_Wuyts']])
			ra_2cat   = concatenate([catalog1_sample['RA'], catalog2_sample['RA']])
			dec_2cat  = concatenate([catalog1_sample['DEC'], catalog2_sample['DEC']])
			photz_2cat = concatenate([catalog1_sample['Photo_z'], catalog2_sample['zbest_candels']])
			id_2cat    = concatenate([catalog1_sample['ID'],catalog2_sample['ID']])
			catalog_2cat = concatenate([zeros(len(catalog1_sample)),ones(len(catalog2_sample))]) #0 = UDS, 1 = COSMOS, 2= GOODS-S
		
			if A_catalog3 != None:
				self.morph = concatenate([A_2catalogs,A_catalog3],axis=1)
				
				self.mass = concatenate([mass_2cat,catalog3_sample['FAST_bc03_lmass']])
				self.sfr  = concatenate([sfr_2cat,catalog3_sample['SFR_Wuyts']])
				self.ra   = concatenate([ra_2cat, catalog3_sample['RA']])
				self.dec  = concatenate([dec_2cat, catalog3_sample['DEC']])
				self.photo_z = concatenate([photz_2cat, catalog3_sample['zbest_candels']])
				self.id    = concatenate([id_2cat,catalog3_sample['ID']])
				twos = ones(len(catalog3_sample))+1
				self.catalog_bool = concatenate([catalog_2cat,twos])  #0 = UDS, 1 = COSMOS, 2= GOODS-S
				
			else:
				self.morph = A_2catalogs
				self.mass = mass_2cat
				self.sfr = sfr_2cat
				self.ra = ra_2cat
				self.dec = dec_2cat
				self.photo_z = photz_2cat
				self.id = id_2cat
				self.catalog_bool = catalog_2cat

				
		else:
			self.morph = A
			self.mass = mass_1cat
			self.sfr = sfr_1cat
			self.ra = ra_1cat
			self.dec = dec_1cat
			self.photo_z = photz_1cat
			self.id = id_1cat
			self.catalog_bool = catalog_1cat
				
		return

def data_multicat(catalog1,  data_fields, catalog2=None, catalog3=None,band='J',zmin=0,zmax=10.0):
    #First catalog
    #Remove bad data (S/N < 4, Missing values, etc.)
    good_data = create_complete_data_morphcat(catalog1,zmin,zmax,band=band)
    catalog1_sample = catalog1[good_data] #selects only galaxies in sample, i.e. M* > 10^10

    #Combine good data into a single data matrix
    A,A_names = select_morph_matrix(catalog1_sample,data_fields,use_flux=False)

    #Second catalog (if exists)
    #Remove bad data (S/N < 4, Missing values, etc.)
    if catalog2 != None:
        good_data_catalog2 = create_complete_data_morphcat(catalog2,zmin,zmax,band=band)
        catalog2_sample = catalog2[good_data_catalog2] #selects only galaxies in sample, i.e. M* > 10^10
        #Combine good data into a single data matrix
        A_catalog2,A_names_catalog2 = select_morph_matrix(catalog2_sample,data_fields,use_flux=False)

        #Catalog2 needs to exist if you want to read catalog3
        if catalog3 != None:
            good_data_catalog3 = create_complete_data_morphcat(catalog3,zmin,zmax,band=band)
            catalog3_sample = catalog3[good_data_catalog3] #selects only galaxies in sample, i.e. M* > 10^10
            A_catalog3,A_names_catalog3 = select_morph_matrix(catalog3_sample,data_fields,use_flux=False)
            
            #Combine all 3 catalogs
            #morph = data_combine(A,catalog1_sample,A_catalog2=A_catalog2,catalog2_sample=catalog2_sample, \
            #                                      A_catalog3=A_catalog3, catalog3_sample=catalog3_sample)
	    morph_class =morph_catalog(A,catalog1_sample,A_catalog2=A_catalog2,catalog2_sample=catalog2_sample, \
                                                  A_catalog3=A_catalog3, catalog3_sample=catalog3_sample)
            #Combine only 2 catalogs
        else:
            #morph = data_combine(A,catalog1_sample,A_catalog2=A_catalog2,catalog2_sample=catalog2_sample)
	    morph_class = morph_catalog(A,catalog1_sample,A_catalog2=A_catalog2,catalog2_sample=catalog2_sample)
    #For when there's only 1 catalog
    else:
        #morph = data_combine(A,catalog1_sample)
	morph_class = morph_catalog(A,catalog1_sample)

    return morph_class

def pca_multicat(A,A_j=None,calculate_vectors=True):
    #PCA
    if calculate_vectors == True:
        pc = ml.PCA(A.T)
    else:
        pc1 = ml.PCA(A_j.T)
        pc = ml.pcV(A.T,pc1.vectors)

    return pc

