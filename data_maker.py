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
	
	if 'zbest_candels' in catalog.names:
		photz_string = 'zbest_candels'
	if 'Photo_z' in catalog.names:
		photz_string = 'Photo_z'
	
	if 'FAST_bc03_lmass' in catalog.names:
		mass_string = 'FAST_bc03_lmass'
	if 'log_Stellar_Mass' in catalog.names:
		mass_string = 'log_Stellar_mass'
	if 'fast_bc03_lmass' in catalog.names:
		mass_string = 'fast_bc03_lmass'

	#print band
	
	good_data = where((isfinite(catalog['SExMAG_%s' % (band)])) &  (catalog['SN_%s' % (band)] >= 4.0) & \
			  (catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax)  &  (catalog['FLAG_%s' % (band)] == 0) & (catalog['C_%s' % (band)] > 0) & \
			  (catalog[mass_string] > 10.0))[0]

	#print catalog
	#print band
	#print "Mag ", len(where((catalog['SExMAG_%s' % (band)] < 40.0) & (catalog['SExMAG_%s' % (band)] > 0.0))[0])
	#print "SN ", len(where(catalog['SN_%s' % (band)] >= 4.0)[0])
	#print "Redshift ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax))[0])
	#print "FLAG ", len(where(catalog['FLAG_%s' % (band)] == 0)[0])
	#print "C ", len(where(catalog['C_%s' % (band)] > 0)[0])
	#print "Mass ", len(where((catalog[mass_string] > 10.0))[0])

	#print "z+SN ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog['SN_%s' % (band)] >= 4.))[0])
	#print "z+C ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog['C_%s' % (band)] > 0))[0])
	#print "z+FLAG ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog['FLAG_%s' % (band)] == 0.0))[0])
	## print "z+Mass ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog[mass_string] > 10.0))[0])
	## print "z+Mass+C ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog[mass_string] > 10.0) & (catalog['C_%s' % (band)] > 0))[0])
	## print "z+Mass+S/N ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog[mass_string] > 10.0) & \
	##(catalog['SN_%s' % (band)] >= 4.))[0])
	## print "z+Mass+FLAG ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog[mass_string] > 10.0) & \
	## 				(catalog['FLAG_%s' % (band)] == 0))[0])
	## print "z+Mass+C+S/N ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog[mass_string] > 10.0) & \
	## 				 (catalog['C_%s' % (band)] > 0) & (catalog['SN_%s' % (band)] >= 4.))[0])
	## print "z+Mass+C+FLAG ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog[mass_string] > 10.0) & \
	## 			     (catalog['C_%s' % (band)] > 0) & (catalog['FLAG_%s' % (band)] == 0))[0])
	## print "z+Mass+S/N+FLAG ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog[mass_string] > 10.0) & \
	## 			       (catalog['SN_%s' % (band)] >= 4.) & (catalog['FLAG_%s' % (band)] == 0))[0])

        ##print "All cuts ", len(good_data)
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

def select_morph_matrix2(morph_cat,fields,use_flux=False):
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

		if A_catalog2 != None:
			A_2catalogs = concatenate([A,A_catalog2],axis=1)
		
			if A_catalog3 != None:
				self.morph = concatenate([A_2catalogs,A_catalog3],axis=1)
			else:
				self.morph = A_2catalogs
		else:
			self.morph = A				
		return

def data_multicat(catalog1,  data_fields, catalog2=None, catalog3=None,band='J',zmin=0,zmax=10.0):
    #First catalog
    #Remove bad data (S/N < 4, Missing values, etc.)
    good_data = create_complete_data_morphcat(catalog1,zmin,zmax,band=band)
    catalog1_sample = catalog1[good_data] #selects only galaxies in sample, i.e. M* > 10^10

    #Combine good data into a single data matrix
    A,A_names = select_morph_matrix2(catalog1_sample,data_fields,use_flux=False)

    #Second catalog (if exists)
    #Remove bad data (S/N < 4, Missing values, etc.)
    if catalog2 != None:
        good_data_catalog2 = create_complete_data_morphcat(catalog2,zmin,zmax,band=band)
        catalog2_sample = catalog2[good_data_catalog2] #selects only galaxies in sample, i.e. M* > 10^10
        #Combine good data into a single data matrix
        A_catalog2,A_names_catalog2 = select_morph_matrix2(catalog2_sample,data_fields,use_flux=False)

        #Catalog2 needs to exist if you want to read catalog3
        if catalog3 != None:
            good_data_catalog3 = create_complete_data_morphcat(catalog3,zmin,zmax,band=band)
            catalog3_sample = catalog3[good_data_catalog3] #selects only galaxies in sample, i.e. M* > 10^10
            A_catalog3,A_names_catalog3 = select_morph_matrix2(catalog3_sample,data_fields,use_flux=False)
            
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

class udf_errorbars:
	def __init__(self,catalog,variables,band='J'):
		magnitude_name = 'SExMAG'+'_'+band+'_UDF'
		udf = where(isfinite(catalog['RA_UDF']) & (catalog['SExMAG_'+band] > 0) & (catalog[magnitude_name] > 0))[0]
		#udf = where(isfinite(catalog['RA_UDF']))[0]
		#['C_H','M20_H','G_H','A_H','M_H','I_H','D_H','R_PET']
		errors = zeros((len(variables),len(udf)))
		i=0
		for parameter in variables:
			parameter_udf = parameter+'_UDF'
			#print parameter, ' ', i
			errors[i,:] = catalog[parameter][udf] - catalog[parameter_udf][udf]
			baderror = where((catalog[parameter][udf] == -99) | (catalog[parameter_udf][udf] == -99))[0]
			errors[i,baderror] = -99.0
			i+=1
			if 'C_' in parameter:
				c_non_negative = where((catalog[parameter][udf] < 0) | (catalog[parameter_udf][udf] < 0))[0]
				errors[i,c_non_negative] = -99.0 

		self.c_error   = errors[0]
		self.m20_error = errors[1]
		self.g_error   = errors[2]
		self.a_error   = errors[3]
		self.m_error   = errors[4]
		self.i_error   = errors[5]
		self.d_error   = errors[6]
		self.rpet_error = errors[7]
		if band == 'J':
			self.mag = catalog['SExMAG_J'][udf]
			self.sn = catalog['SN_J'][udf]
		if band == 'H':
			self.mag = catalog['SExMAG_H'][udf]
			self.sn = catalog['SN_H'][udf]
				
		return

def completeness_morphcat(catalog,zmin=0,zmax=10,band='J',field='UDS'):
	
	if 'zbest_candels' in catalog.names:
		photz_string = 'zbest_candels'
	if 'Photo_z' in catalog.names:
		photz_string = 'Photo_z'
	
	if 'FAST_bc03_lmass' in catalog.names:
		mass_string = 'FAST_bc03_lmass'
	if 'log_Stellar_Mass' in catalog.names:
		mass_string = 'log_Stellar_mass'
	
	good_data = where((isfinite(catalog['SExMAG_%s' % (band)])) &  (catalog['SN_%s' % (band)] >= 4.0) & \
			  (catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax)  &  (catalog['FLAG_%s' % (band)] == 0) & (catalog['C_%s' % (band)] > 0) & \
			  (catalog[mass_string] > 10.0))[0]

	
	#print field
	print band
	print "N Morph ", len(catalog['SExMAG_%s' % (band)])
	print "FLAG ", len(where(catalog['FLAG_%s' % (band)] == 0)[0])
	print "C ", len(where(catalog['C_%s' % (band)] > 0)[0])
	print "SN ", len(where(catalog['SN_%s' % (band)] >= 4.0)[0])
	print "Mass ", len(where((catalog[mass_string] > 10.0))[0])
	print "Redshift ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax))[0])
	print "Mag Finite ", len(where((isfinite(catalog['SExMAG_%s' % (band)])))[0])
	print "Mag < 24.5 ", len(where((catalog['SExMAG_%s' % (band)] < 24.5) & (catalog['SExMAG_%s' % (band)] > 0.0))[0])

	print
	print "z+Mag Finite ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & isfinite(catalog['SExMAG_%s' % (band)]))[0])
	print "z+Bright ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog['SExMAG_%s' % (band)] < 24.5) & \
					 (catalog['SExMAG_%s' % (band)] > 0.0))[0])
	print "z+C ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog['C_%s' % (band)] > 0))[0])
	print "z+SN ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog['SN_%s' % (band)] >= 4.))[0])
	print "z+Mass ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog[mass_string] > 10.0))[0])
	print "z+FLAG ", len(where((catalog[photz_string] > zmin) &  (catalog[photz_string] < zmax) & (catalog['FLAG_%s' % (band)] == 0.0))[0])
	

        print "All cuts ", len(good_data)
	return good_data

def redshift_sample(catalog, band='J',zmin=0,zmax=10.0):
    #First catalog
    #Remove bad data (S/N < 4, Missing values, etc.)
    good_data = create_complete_data_morphcat(catalog,zmin,zmax,band=band)
    catalog_sample = catalog[good_data]
    return catalog_sample

def data_multicat_full(catalog1,catalog2=None, catalog3=None,band='J',zmin=0,zmax=10.0):
    #First catalog
    #Remove bad data (S/N < 4, Missing values, etc.)
    good_data = create_complete_data_morphcat(catalog1,zmin,zmax,band=band)
    catalog1_sample = catalog1[good_data] #selects only galaxies in sample, i.e. M* > 10^10

    #Combine good data into a single data matrix
    #A,A_names = select_morph_matrix2(catalog1_sample,data_fields,use_flux=False)

    #Second catalog (if exists)
    #Remove bad data (S/N < 4, Missing values, etc.)
    if catalog2 != None:
        good_data_catalog2 = create_complete_data_morphcat(catalog2,zmin,zmax,band=band)
        catalog2_sample = catalog2[good_data_catalog2] #selects only galaxies in sample, i.e. M* > 10^10
        #Combine good data into a single data matrix
        #catalog2,names_catalog2 = select_morph_matrix2(catalog2_sample,data_fields,use_flux=False)

        #Catalog2 needs to exist if you want to read catalog3
        if catalog3 != None:
            good_data_catalog3 = create_complete_data_morphcat(catalog3,zmin,zmax,band=band)
            catalog3_sample = catalog3[good_data_catalog3] #selects only galaxies in sample, i.e. M* > 10^10
	    #Combine all 3 catalogs
	    morph_class =morphcat(catalog1_sample,catalog2=catalog2_sample,catalog3=catalog3_sample)
            #Combine only 2 catalogs
        else:
            
	    morph_class = morphcat(catalog1_sample,catalog2=catalog2_sample)
    #For when there's only 1 catalog
    else:
        #morph = data_combine(A,catalog1_sample)
	morph_class = morphcat(catalog1_sample)
    
    return morph_class

def morphcat(catalog1, catalog2=None,catalog3=None):

	new_dict = {}
	catalog1 = capitalize_keys(catalog1)
	catalog1_names = catalog1.keys()
	if catalog2 != None:
		catalog2 = capitalize_keys(catalog2)
		catalog2_names = catalog2.keys()
		
	if catalog3 != None:
		catalog3 = capitalize_keys(catalog3)
		catalog3_names = catalog3.keys()

	#Get list of all field names from all catalogs

	if catalog2 != None:
		if catalog3 != None:
			all_parameters = [catalog1.keys()+catalog2.keys()+catalog3.keys()]
		else:
			all_parameters = [catalog1.keys()+catalog2.keys()]
	else:
		all_parameters = [catalog1.keys()]

	all_parameters_unique = unique(array(all_parameters)) #Remove duplicate field names from list


	#Create and update new dictionary that is the concantenation of all constituent catalogs
	for param in all_parameters_unique:
		if param in catalog1_names:
			#print catalog1.names[where(catalog1.names == param)[0]]
			#print param, " exists"
			new_dict[param] = catalog1[param].tolist()
		else:
			#print param, " doesn't exist"
			new_param_default = -99.0*ones(len(catalog1[catalog1.keys()[0]]))
			new_dict[param] = new_param_default.tolist() #Create list of -99.0 for when parameter is missing
		if catalog2 != None:
			for ngal2 in range(len(catalog2[catalog2.keys()[0]])):
				if param in catalog2_names:
					new_dict[param].append(catalog2[param][ngal2])
				else:
					new_dict[param].append(-99)

			if catalog3 != None:
				for ngal3 in range(len(catalog3[catalog3.keys()[0]])):
					if param in catalog3_names:
						new_dict[param].append(catalog3[param][ngal3])
					else:
						new_dict[param].append(-99)
		new_dict[param] = array(new_dict[param])
	return new_dict

def fixConcentration(catalog,band='J',zmin=1.36,zmax=1.97):
	new_catalog = catalog
	badc = where((catalog['Photo_z'] > zmin) &  (catalog['Photo_z'] < zmax) & (catalog['log_stellar_mass'] > 10.0) & (catalog['C_%s' % (band)] < 0))[0]
	r80 = catalog['R80_%s' % (band)][badc]*0.06 #Converts pixels to arcseconds
	new_concentration = 5*log10(r80/0.08) #0.08 Is size of PSF, lower limit on C
	new_catalog['C_%s' % (band)][badc] = new_concentration
	#print len(badc)
	return new_catalog[badc]

def capitalize_keys(d):
	#Convert fits rec to dictionary, while also capitalizing keys/names
	result = {}
	for key in d.names:
		upper_key = key.upper()
		result[upper_key] = d[key]
	return result

def full_z_sample(catalog,zmin=0,zmax=10,band='J'):
	
	if 'zbest_candels' in catalog.names:
		photz_string = 'zbest_candels'
	if 'Photo_z' in catalog.names:
		photz_string = 'Photo_z'
	
	if 'FAST_bc03_lmass' in catalog.names:
		mass_string = 'FAST_bc03_lmass'
	if 'log_Stellar_Mass' in catalog.names:
		mass_string = 'log_Stellar_mass'
	
	good_data = where((isfinite(catalog['SExMAG_%s' % (band)])) &  (catalog['SN_%s' % (band)] >= 4.0) & \
			  (catalog['FLAG_%s' % (band)] == 0) & (catalog['C_%s' % (band)] > 0) & \
			  (catalog[mass_string] > 10.0))[0]
	
	return good_data

def full_mass_sample(catalog,zmin=0,zmax=10,band='J'):
	
	if 'zbest_candels' in catalog.names:
		photz_string = 'zbest_candels'
	if 'Photo_z' in catalog.names:
		photz_string = 'Photo_z'
	
	if 'FAST_bc03_lmass' in catalog.names:
		mass_string = 'FAST_bc03_lmass'
	if 'log_Stellar_Mass' in catalog.names:
		mass_string = 'log_Stellar_mass'
	
	good_data = where((isfinite(catalog['SExMAG_%s' % (band)])) &  (catalog['SN_%s' % (band)] >= 4.0) & \
			  (catalog[photz_string] > zmin)  & (catalog[photz_string] < zmax)  &  (catalog['FLAG_%s' % (band)] == 0) & (catalog['C_%s' % (band)] > 0))[0]
	
	return good_data

def zsample_multicat(catalog1,catalog2=None, catalog3=None,band='J',zmin=0,zmax=10.0):
    #First catalog
    #Remove bad data (S/N < 4, Missing values, etc.)
    good_data = full_z_sample(catalog1,zmin,zmax,band=band)
    catalog1_sample = catalog1[good_data] #selects only galaxies in sample, i.e. M* > 10^10

    #Combine good data into a single data matrix
    #A,A_names = select_morph_matrix2(catalog1_sample,data_fields,use_flux=False)

    #Second catalog (if exists)
    #Remove bad data (S/N < 4, Missing values, etc.)
    if catalog2 != None:
        good_data_catalog2 = full_z_sample(catalog2,zmin,zmax,band=band)
        catalog2_sample = catalog2[good_data_catalog2] #selects only galaxies in sample, i.e. M* > 10^10
        #Combine good data into a single data matrix
        #catalog2,names_catalog2 = select_morph_matrix2(catalog2_sample,data_fields,use_flux=False)

        #Catalog2 needs to exist if you want to read catalog3
        if catalog3 != None:
            good_data_catalog3 = full_z_sample(catalog3,zmin,zmax,band=band)
            catalog3_sample = catalog3[good_data_catalog3] #selects only galaxies in sample, i.e. M* > 10^10
	    #Combine all 3 catalogs
	    morph_class =morphcat(catalog1_sample,catalog2=catalog2_sample,catalog3=catalog3_sample)
            #Combine only 2 catalogs
        else:
            
	    morph_class = morphcat(catalog1_sample,catalog2=catalog2_sample)
    #For when there's only 1 catalog
    else:
        #morph = data_combine(A,catalog1_sample)
	morph_class = morphcat(catalog1_sample)
    
    return morph_class

def mass_sample_multicat(catalog1,catalog2=None, catalog3=None,band='J',zmin=0,zmax=10.0):
    #First catalog
    #Remove bad data (S/N < 4, Missing values, etc.)
    good_data = full_mass_sample(catalog1,zmin,zmax,band=band)
    catalog1_sample = catalog1[good_data] #selects only galaxies in sample, i.e. M* > 10^10

    #Combine good data into a single data matrix
    #A,A_names = select_morph_matrix2(catalog1_sample,data_fields,use_flux=False)

    #Second catalog (if exists)
    #Remove bad data (S/N < 4, Missing values, etc.)
    if catalog2 != None:
        good_data_catalog2 = full_mass_sample(catalog2,zmin,zmax,band=band)
        catalog2_sample = catalog2[good_data_catalog2] #selects only galaxies in sample, i.e. M* > 10^10
        #Combine good data into a single data matrix
        #catalog2,names_catalog2 = select_morph_matrix2(catalog2_sample,data_fields,use_flux=False)

        #Catalog2 needs to exist if you want to read catalog3
        if catalog3 != None:
            good_data_catalog3 = full_mass_sample(catalog3,zmin,zmax,band=band)
            catalog3_sample = catalog3[good_data_catalog3] #selects only galaxies in sample, i.e. M* > 10^10
	    #Combine all 3 catalogs
	    morph_class =morphcat(catalog1_sample,catalog2=catalog2_sample,catalog3=catalog3_sample)
            #Combine only 2 catalogs
        else:
            
	    morph_class = morphcat(catalog1_sample,catalog2=catalog2_sample)
    #For when there's only 1 catalog
    else:
        #morph = data_combine(A,catalog1_sample)
	morph_class = morphcat(catalog1_sample)
    
    return morph_class
