#!/usr/bin/env python

"""
This script will combine the Multi-wavelength catalogs, Photo-z/Stellar Mass catalogs and Gini-M20 morphology catalogs into a single catalog that
will then be run with PCA to determine the important dimensions
"""

#Algorithm
#1. Take ascii-tables and convert to SExtractor format
#2. Grab dictionary key names and check if some overlap, if so rename to band specific e.g. + "_J"
#3. Match by IDs if they are consistent, otherwise match by RA/Dec
#4. Compile all data into 1 dictionary
#5. Write Dictionary to FITS file (?) Do I need this?

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
import astropysics.coords as ac

import glob

#My Personally created modules
## import data_maker as dm
## import plot_maker as pm
## import machine_learning as ml
## import convex_check as cvx
## import stamps as st
## import stat_tools as stat

def internalMatch(directory='',outputfile = 'table.fits', match_by_id=True):

    new_dict = {}
    columns = []
    column_names = []
    ra = []
    dec = []
    
    filenames=glob.glob(directory+"*.txt")
    if match_by_id==True:
        for filename in filenames:
            print filename
            data_class = sextractor(filename)
            data_dict = data_class.__dict__
            parameters = data_dict.keys()
            for param in parameters:
                if param[0] != '_':
                    #If Tables are already in correct order, no matching needed

                    new_dict[param] = data_dict[param]

                    #Determine the format type for column, needed to write FITs file
                    if type(data_dict[param][0]) == str:
                        fmt='A'
                    elif type(data_dict[param][0]) == int32:
                        fmt='J'
                    else:
                        fmt='D'

                    print param

                    if param not in column_names:
                        col1 = pyfits.Column(name=param, format=fmt, array=data_dict[param])
                        column_names.append(param) # = column_names + [param]
                        columns+=[col1] # = columns + [col1]
    #If Tables are not in same order, or IDs not the same
    else:
        dict_list = []
        all_parameters = []
        for filename in filenames:
            print filename
            data_class = sextractor(filename)
            data_dict = data_class.__dict__
            dict_list.append(data_dict) #Append next dictionary
            ## parameters = data_dict.keys()
            ## all_parameters.extend(parameters)
            ## unique(all_parameters)
            if 'photom' in filename:
                idx = string.find(filename,'photom')
                suffix = '_'+str(filename[idx-1])
                morph_parameters = data_dict.keys()
            

        from astropy.coordinates import ICRS
        from astropy import units as u
        from astropy.coordinates import match_coordinates_sky

        new_dict = dict_list[0] #.__dict__ #Initialize new dictionary with first read-in SExtractor file
        all_parameters = new_dict.keys()
        for ncat in range(len(filenames)-1):
            if 'alpha_gr_dec_order' in new_dict.keys():
                ra_string1 = 'alpha_gr_dec_order'
                dec_string1 = 'delta_gr_dec_order'
            if 'ra' in new_dict.keys():
                ra_string1 = 'ra'
                dec_string1 = 'dec'

            if 'alpha_gr_dec_order' in dict_list[ncat+1].keys():
                ra_string2 = 'alpha_gr_dec_order'
                dec_string2 = 'delta_gr_dec_order'
            if 'ra' in  dict_list[ncat+1].keys():
                ra_string2 = 'ra'
                dec_string2 = 'dec'
            
            arcsec = 0.000277777778 #Degrees per arcsec

            c1 = ICRS(new_dict[ra_string1],  new_dict[dec_string1],  unit=(u.degree, u.degree))
            c2 = ICRS(dict_list[ncat+1][ra_string2], dict_list[ncat+1][dec_string2], unit=(u.degree, u.degree))
            
            idx, d2d, d3d = match_coordinates_sky(c1, c2) #idx has indicies of c2, with shape of c1, but with no separation requirement
            idx1 = where(d2d.arcsec < 1.0)[0] #Degree separation, d2d.arcsec, Indicies for c1, less than 1 arcsec
            #print len(idx1)
            #print max(idx1)

            idx2 = idx[idx1] #Indicies for c2

            #Add keys from additional catalogs to be matched to
            parameters = dict_list[ncat+1].keys()
            all_parameters.extend(parameters)
            unique(all_parameters)

            for param in all_parameters:
                if (not param[0] == '_'):# or (not param == 'redshift'):
                    print param
                    if param in new_dict.keys():
                        #print len(new_dict[param])
                        new_dict[param] = new_dict[param][idx1] #Select indicies that match to catalog2
                    elif param in dict_list[ncat+1].keys():
                        new_dict[param] = dict_list[ncat+1][param][idx2] #For when the dictionary didn't have the parameter yet, take from second catalog
                    else:
                        new_dict[param] = -99*ones(len(idx2))

                    #Determine the format type for column, needed to write FITs file
                    if type(new_dict[param][0]) == str:
                        fmt='A'
                    elif type(new_dict[param][0]) == int32:
                        fmt='J'
                    else:
                        fmt='D'

                    if param not in column_names:
                        if param in morph_parameters:
                            psuff = param+suffix
                            col1 = pyfits.Column(name=psuff.upper(), format=fmt, array=new_dict[param])
                        else:
                            col1 = pyfits.Column(name=param, format=fmt, array=new_dict[param])
                        column_names.append(param)
                        columns+=[col1]

    tbhdu = pyfits.new_table(columns)
    print "Writing ", outputfile
    tbhdu.writeto(directory+outputfile,clobber=True)
    return new_dict

