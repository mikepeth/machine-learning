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
#import plot_maker as pm
import machine_learning as ml
import convex_check as cvx
import stamps as st
        
def stamp_collection(data,label=None,band='f125w'):
    if label == None:
        ward = Ward(n_clusters=10).fit(X)
        label = ward.labels_

    if band=='f160w':
        img = '/Users/Mike/Dropbox/candels/mosaics/uds/uds_2epoch_wfc3_f160w_060mas_v0.3_drz.fits' #F160W
    elif band=='f125w':
        img = '/Users/Mike/Dropbox/candels/mosaics/uds/uds_2epoch_wfc3_f125w_060mas_v0.3_drz.fits' #F125W
    elif band=='f814w':
        img = '/Users/Mike/Dropbox/candels/mosaics/uds/uds_2epoch_acs_f814w_060mas_v0.3_drz.fits' #F814W
    else:
        print "Please use f160w, f125w or f814w"
        return

    #RA, Dec
    ra = data.ra
    dec = data.dec
        
    dest = '/Users/Mike/Dropbox/pca/code/uds/plots/stamp/'+band+'/'
    for n in range(clusters):
        labeln = where(label == n)[0]
        d=dest+'group'+str(n)+'/'
        if not os.path.exists(d):
            os.makedirs(d)
        #for i in range(len(labeln)):
        if len(labeln) > 10:
            short_num = 10
        else:
            short_num = len(labeln)
        for i in range(short_num): #Makes at most 10 stamps per group
            postage = d+'galID_'+str(labeln[i])+'.png'
            radec = array([[ra[labeln][i],dec[labeln][i]]], float_)
            st.stamp_grey(radec,mosaic=img,outfile=postage)
    return

def rgb_stamps(catalog,X,clusters=8,group_start=0):
    ward = Ward(n_clusters=clusters).fit(X)
    label = ward.labels_
    rimg = '/Users/Mike/Dropbox/candels/mosaics/uds/uds_2epoch_wfc3_f160w_060mas_v0.3_drz.fits' #F160W
    gimg = '/Users/Mike/Dropbox/candels/mosaics/uds/uds_2epoch_wfc3_f125w_060mas_v0.3_drz.fits' #F125W
    bimg = '/Users/Mike/Dropbox/candels/mosaics/uds/uds_2epoch_acs_f814w_060mas_v0.3_drz.fits' #F814W

    import stamps as st
    rgb_list = [rimg,gimg,bimg] #Red, Green, Blue
    scales= array([1.3,1.0,0.8])   #relative scaling of RGB images


    import stamps as st
    dest = '/Users/Mike/Dropbox/pca/code/uds/plots/stamp/rgb/'
    for n in range(group_start,clusters):
        labeln = where(label == n)[0]
        d=dest+'group'+str(n)+'/'
        if not os.path.exists(d):
            os.makedirs(d)
        for i in range(len(labeln)):
            rgb_list2 = []
            for band in range(len(rgb_list)):
                #png_file='test'+str(band)+'.png'
                fits_file='test'+str(band)+'.fits'
                st.stamp_grey(catalog, num=labeln[i], mosaic=rgb_list[band],output_fits=fits_file,save_to_png=False,scale=scales[band])
                rgb_list2 = rgb_list2+[fits_file]
            postage = d+'galID_'+str(cat_sample['ID'][labeln][i])+'_rgb.png'
            print "Saving ", postage
            st.save_rgb_stamp(rgb_list2,outfile=postage,vmin=0,vmax=0.4,vmid=0.2)
    return
    

def demographics(catalog,X,clusters=8,textfile='pc_hclust_viz_class.txt'):
    ward = Ward(n_clusters=clusters).fit(X)
    label = ward.labels_
    f=open(textfile, 'w')
    for n in range(clusters):
        ntype = where(label == n)[0]
        f.write( "Type %s \n" % (str(n)))
        f.write( "Number %s \n" % str(len(ntype)))
        pure_disk_ntype = where((catalog['disk'][ntype] > 0.6) &(catalog['spheroid'][ntype] < 0.3))[0]
        pure_bulge_ntype = where((catalog['disk'][ntype] < 0.3) &(catalog['spheroid'][ntype] > 0.6))[0]
        bulge_disk_ntype = where((catalog['disk'][ntype] > 0.6)&(catalog['spheroid'][ntype] > 0.6))[0]

        clumpy_ntype  = where((catalog['c2p2'][ntype] > 0.6) | (catalog['c2p1'][ntype] > 0.6) | (catalog['c2p0'][ntype] > 0.6) | \
                        (catalog['c1p2'][ntype] > 0.6) | (catalog['c1p1'][ntype] > 0.6) | (catalog['c1p0'][ntype] > 0.6))[0]

        sph_ntype = where(catalog['spheroid'][ntype] > 0.6)[0]
        ps_ntype = where(catalog['ps'][ntype] > 0.6)[0]
        disk_ntype = where(catalog['disk'][ntype] > 0.6)[0]
        irregular_ntype = where(catalog['irr'][ntype] > 0.5)[0]

        merger_ntype = where((catalog['merger'][ntype] > 0.5) | (catalog['int1'][ntype] > 0.5) | (catalog['int2'][ntype] > 0.5))[0] #
        #total_mergers = where((catalog['merger'] > 0.5) | (catalog['int1'] > 0.5) | (catalog['int2'] > 0.5))[0] #

        #UVJ Color QG vs SFG
        ssfr = log10(catalog['SFR'][ntype])-catalog['log_stellar_mass'][ntype]
        quiescent_ntype = where(ssfr < -11)[0]
        sf_ntype = where(ssfr > -11)[0]

        f.write( "Pure disk %s \n" % str(len(pure_disk_ntype)))
        f.write( "Pure spheroid %s \n" % str(len(pure_bulge_ntype)))
        f.write( "Bulge+Disk %s \n" % str(len(bulge_disk_ntype)))
        f.write( "Clumpy %s \n" % str(len(clumpy_ntype)))
        f.write( "Spheroids (any) %s \n" % str(len(sph_ntype)))
        f.write( "Disks (any) %s \n" % str(len(disk_ntype)))
        f.write( "Irregular %s \n" % str(len(irregular_ntype)))
        f.write( "Point Source %s \n" % str(len(ps_ntype)))
        f.write( "Mergers (mergers+interactions+companions) %s \n" % str(len(merger_ntype)))
        f.write( "Quiescent %s \n" % str(len(quiescent_ntype)))
        f.write( "Star Forming %s \n" % str(len(sf_ntype)))
        f.write("\n")
    f.close()

    return

def cluster_plots(mass_sfr,X,n=10,ssfr_check=False):

    f = open('./cluster/cluster_j_candels_demographics.txt', 'w')
    print n, " Clusters"
    clusters = n
    f.write(str(n)+ " Clusters")
    f.write("\n")
    ward = Ward(n_clusters=clusters).fit(X[:,0:3])
    label = ward.labels_

    #3D Plot
    fig = pl.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(label):
        ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
                  'o', color=pl.cm.jet(float(l) / np.max(label + 1)))
    pl.title('Hierarchical Clustering, %s Clusters' % (str(clusters)))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_ylim(-5,5)
    pl.show()
    savefig('./cluster/clustering_pc_j_candels_nclust%s.pdf' %(str(clusters)))
 
    #sSFR - Mass
    if ssfr_check==True:
        ssfr = log10(mass_sfr[1])-mass_sfr[0] #log(SFR) - log(Mass)
        mass = mass_sfr[0]
        cm = matplotlib.cm.get_cmap('jet')

        for grp in range(n):
            fig = figure()
            group = where(ward.labels_ == grp)[0]
            plot(mass[group],ssfr[group],'o',color=pl.cm.jet(float(grp) / np.max(n + 1)))
    #        colorbar()
            plot(arange(9,13,0.1),zeros(len(arange(9,13,0.1)))-11,'--r')
            xlim(9.9,12.0)
            ylim(-16,-6)
            xlabel('Log Mass')
            ylabel('sSFR')
            title('Group %s, N=%s'% (str(grp),str(len(group))))
            savefig('./cluster/clustering_ssfr_mass_j_candels_nclust%s_group%s.pdf' %(str(clusters),str(grp)))
            close('all')

        #Quiescent demographics for each bin
        qg_tot = len(where(ssfr < -11)[0])
        sfg_tot = len(where(ssfr > -11)[0])

        for grp in np.unique(label):
            n_group = len(ssfr[label == grp])
            quiescent = where(ssfr[label == grp] < -11)[0]
            sfg = where(ssfr[label == grp] > -11)[0]
            output="Group "+str(grp)+": "+str(n_group)+" Galaxies in Group, "+str(round(len(quiescent)/float(n_group)*100,1))+ \
                    " QG %, "+str(round(len(sfg)/float(n_group)*100,1))+" SFG %"
            f.write(output)
            f.write("\n")
    f.close()
    return

def cluster_plots_1field(data,X,n=10,band='j',label=None,ssfr_check=False):

    clusters = n
    if label==None:
        print n, " Clusters"
        
        ward = Ward(n_clusters=clusters).fit(pc.X)
        label = ward.labels_

    #Band labels
    if band == 'j':
        zmin = 1.36
        zmax = 1.97
        full_band = 'F125W'
    if band == 'h':
        zmin = 2.0
        zmax = 2.57
        full_band = 'F160W'
    if band == 'i':
        zmin = 0.5
        zmax = 1.05
        full_band = 'F814W'
    if band == 'v':
        zmin = 0.0
        zmax = 0.5
        full_band = 'F606W'

    #3D Plot
    fig = pl.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for l in np.unique(label):
        ax.plot3D(pc.X[label == l, 0], pc.X[label == l, 1], pc.X[label == l, 2],
                  'o', color=pl.cm.jet(float(l) / np.max(label + 1)))
    pl.title('%s, %s<z<%s' % (full_band,str(zmin),str(zmax)))
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_xlim(-8,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    pl.show()
    savefig('/Users/Mike/Dropbox/pca/paper/figures/clustering_pc_%s_candels_nclust%s.pdf' %(band,str(clusters)))
 
    #sSFR - Mass
    if ssfr_check==True:
        ssfr = log10(data.sfr)-data.mass #log(SFR) - log(Mass)
        mass = data.mass #log(Mass)
        cm = matplotlib.cm.get_cmap('jet')

        for grp in range(-1,n,1):
            
            group = where(label == grp)[0]
            if len(group) > 3:
                fig = figure()
                plot(mass[group],ssfr[group],'o',color=pl.cm.jet(float(grp) / np.max(n + 1)))
                plot(arange(9,13,0.1),zeros(len(arange(9,13,0.1)))-11,'--r')
                xlim(9.9,12.0)
                ylim(-16,-6)
                xlabel('Log Mass')
                ylabel('sSFR')
                title('Group %s, N=%s'% (str(grp),str(len(group))))
                #savefig('/Users/Mike/Dropbox/pca/paper/figures/clustering_ssfr_mass_%s_candels_nclust%s_group%s.pdf' %(band,str(clusters),str(grp)))
            close('all')

    return

def cluster_plots_ssfr_2fields(mass_sfr1,X1,mass_sfr2,X2,n=10,label1=None,label2=None,bands='jh',second_band='F814W'):
    #sSFR - Mass
    ssfr1 = log10(mass_sfr1[1])-mass_sfr1[0] #log(SFR) - log(Mass)
    mass1 = mass_sfr1[0]

    ssfr2 = log10(mass_sfr2[1])-mass_sfr2[0] #log(SFR) - log(Mass)
    mass2 = mass_sfr2[0]
    cm = matplotlib.cm.get_cmap('jet')

    for grp in range(-1,n,1):
        group1 = where(label1 == grp)[0]
        group2 = where(label2 == grp)[0]
        if len(group1) > 3:
            fig = figure()
            plot(mass1[group1],ssfr1[group1],'ok',label='F125W')#,color=pl.cm.jet(float(grp) / np.max(n + 1)))
            plot(mass2[group2],ssfr2[group2],'*r',markersize=10,label=second_band)#,color=pl.cm.jet(float(grp) / np.max(n + 1)))
            legend(loc=4)
            plot(arange(9,13,0.1),zeros(len(arange(9,13,0.1)))-11,'--r')
            xlim(9.9,12.0)
            ylim(-16,-6)
            xlabel('Log Mass')
            ylabel('sSFR')
            title('Group %s, N=%s'% (str(grp),str(len(group1)+len(group2))))
            savefig('./cluster/clustering_ssfr_mass_%s_candels_nclust%s_group%s.pdf' %(bands,str(n),str(grp)))
            #close('all')
            
    return

def full_stamp_catalog(A,pc,filename,label=None):
    #Choose only galaxies from UDS
    uds_only = where(A.catalog_bool == 0)[0]

    f = open(filename,'w')
    f.write('# 1 RA \n')
    f.write('# 2 DEC \n')
    for dim in range(len(pc[0])):
        f.write('# %s PC%s \n' %(str(dim+3),str(dim+1)))
    f.write('# 10 C_J \n')
    f.write('# 11 M20_J \n')
    f.write('# 12 G_J \n')
    f.write('# 13 A_J \n')
    f.write('# 14 M_J \n')
    f.write('# 15 I_J \n')
    f.write('# 16 D_J \n')
    f.write('# 17 log_mass \n')
    f.write('# 18 Photz \n')
    f.write('# 19 SFR \n')
    f.write('# 20 ID \n')
    f.write('# 21 CATALOG uds=0, cosmos=1, goods-s=2')
    if label != None:
        f.write('# 21 group \n')
    for n_gal in range(len(pc)): #uds_only:
        f.write('%s \t' %(A.ra[n_gal])) #RA
        f.write('%s \t' %(A.dec[n_gal])) #DEC
        for dim in range(len(pc[0])):
            f.write('%s \t' %(pc[n_gal][dim]))
        f.write('%s \t' %(A.morph[0][n_gal])) #C_J
        f.write('%s \t' %(A.morph[1][n_gal])) #['M20_J']
        f.write('%s \t' %(A.morph[2][n_gal])) #['G_J']
        f.write('%s \t' %(A.morph[3][n_gal])) #['A_J']
        f.write('%s \t' %(A.morph[4][n_gal])) #['M_J']
        f.write('%s \t' %(A.morph[5][n_gal])) #['I_J']
        f.write('%s \t' %(A.morph[6][n_gal])) #['D_J']
        f.write('%s \t' %(A.mass[n_gal])) #Log_mass
        f.write('%s \t' %(A.photo_z[n_gal])) #Photo-z
        f.write('%s \t' %(A.sfr[n_gal])) #SFR
        f.write('%s \t' %(A.id[n_gal])) #ID
        if label != None:
            f.write('%s \t' %(label[n_gal]))
        f.write('%s \t' %(A.catalog_bool[n_gal])) #CATALOG BOOL
        f.write('\n')
    f.close()
    return

def simple_demographics(label):
    ngal = len(label)
    for n in range(-1,10,1):
        print "Group ", n, " N=", ngal
        labeln = where(label == n)[0]
        ngal_class_percentage = float(len(labeln))/ngal*100
        print ngal_class_percentage, "%"
    return
    

#Global Variables
data_fields_j = ['C_J','M20_J','G_J','A_J','M_J','I_J','D_J']
data_fields_h = ['C_H','M20_H','G_H','A_H','M_H','I_H','D_H']
data_fields_i = ['C_I','M20_I','G_I','A_I','M_I','I_I','D_I']
data_fields_v = ['C_V','M20_V','G_V','A_V','M_V','I_V','D_V']
#bands = ['Vband','Iband','Jband','Hband']
#redshift_range_v = [0.0,0.5]
#redshift_range_i = [0.5,1.05]
#redshift_range_j = [1.36,1.97]
#redshift_range_h = [2.0,2.57]
#scatter_parameters = ['mass','sfr','m20','photz','gini','disk','irr','mbrf','c','d','merger']

#Main Program
#Read-in Catalogs
#UDS Catalogs
uds_hdu = pyfits.open('/Users/Mike/Dropbox/candels/morph/uds/UDS_mw_morph_viz_pz.fits') #UDS
uds = uds_hdu[1].data

cos_hdu = pyfits.open('/Users/Mike/Dropbox/candels/morph/cosmos/cosmos_rainbow_mwcat_zmass_morph.fits') #COS
cos = cos_hdu[1].data

goodss_hdu = pyfits.open('/Users/Mike/Dropbox/candels/morph/goodss/goodss_rainbow_mwcat_zmass_morph.fits') #GOODS-S
goodss = goodss_hdu[1].data

#Calculate PC's
A_j = dm.data_multicat(uds,data_fields_j, catalog2=cos,catalog3=goodss,band='J',zmin=1.36,zmax=1.97)
A_h = dm.data_multicat(uds,data_fields_h, catalog2=cos,catalog3=goodss,band='H',zmin=1.99,zmax=2.57)
A_i = dm.data_multicat(uds,data_fields_i, catalog2=cos, band='I',zmin=0.51,zmax=1.05)
A_v = dm.data_multicat(uds,data_fields_v, catalog2=cos, band='V',zmin=0.0,zmax=0.51)

pc   = dm.pca_multicat(A_j.morph,calculate_vectors=True)
pc_h = dm.pca_multicat(A_h.morph,A_j=A_j.morph,calculate_vectors=False)
pc_i = dm.pca_multicat(A_i.morph,A_j=A_j.morph,calculate_vectors=False)
pc_v = dm.pca_multicat(A_v.morph,A_j=A_j.morph,calculate_vectors=False)

#Visual Demographics
#demographics(cat_sample,pc.X,textfile='pc_hclust_viz_class_nclust10.txt',clusters=10)

#RGB Images
#rgb_stamps(cat_sample,pc.X)

#Tables

#Clustering Plots
#cluster_plots(mass_sfr_j,pc.X,n=10)

#Data table


clusters=10
ward = Ward(n_clusters=clusters).fit(pc.X)
label_j = ward.labels_
label_j[where((label_j == 5) | (label_j == 7) | (label_j == 8))[0]]=-1 #regroup all outliers (5,7,8) into group=-1

#Labels for F160W 2<z<2.5
print 'Labels for F160W 2<z<2.5'
#label_h = cvx.pc_convexhull(pc_h.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

print 'Labels for F814W 0.5<z<1.05, Doesnt include GOODS-S'
#Labels for F814W 0.5<z<1.05, Doesn't include GOODS-S
#label_i = cvx.pc_convexhull(pc_i.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

print 'Labels for F606W 0.0<z<0.5, Doesnt include GOODS-S'
#Labels for F814W 0.0<z<0.5, Doesn't include GOODS-S
#label_v = cvx.pc_convexhull(pc_v.X,pc.X,radius=1e-6) #Radius of 1e-6 gives lowest number of mistaken labels

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
#stamp_collection(A_j,label=label_j,band='f125w')
#stamp_collection(A_i,label=label_i,band='f814w')
#stamp_collection(A_h,label=label_h,band='f160w')

print "F125W 1.36<z<1.97"
#simple_demographics(label_j)
print "F160W 2<z<2.5"
#simple_demographics(label_h)
print "F814W 0.5<z<1.0"
#simple_demographics(label_i)
print "F125W 0.0<z<0.5"
#simple_demographics(label_v)

#Figures
#F125W
cluster_plots_1field(A_j,pc.X,label=label_j,n=10,band='j',ssfr_check=True)

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
