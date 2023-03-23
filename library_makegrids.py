import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import constants as cst
import speasy as spz
from datetime import timedelta, datetime, timezone
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
from scipy.ndimage import gaussian_filter as gf

from scipy.optimize import fsolve
from math import radians as rad
from math import degrees as deg

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

from functools import partial
from multiprocessing import Pool

import sys
sys.path.append('./space')
sys.path.append('Python_functions')
sys.path.append('./')
#sys.path.insert(0,'Python_functions/py_tsyganenko/build/')

from space.models import planetary as smp

from space import smath as sm
from space.coordinates import coordinates as scc
from space import utils as su
import py_tsyganenko.Geopack as gp
import py_tsyganenko.Models as tm
from scipy.integrate import solve_ivp

from skimage.feature import hessian_matrix,hessian_matrix_eigvals
from scipy.interpolate import griddata

from sklearn.neighbors import KNeighborsRegressor
import ML_boundaries_models as bm
import utilities as uti

import speasy as spz
import datetime


msh = smp.Magnetosheath(magnetopause='mp_shue1998', bow_shock ='bs_jelinek2012')
import shear_maps as smap
import smap_plot as smapplt
#msh = smp.Magnetosheath(magnetopause='mp_sibeck1991', bow_shock ='bs_jelinek2012')

def fig_nx2_plots(n=2,norm=Normalize(-60,60),cmap='seismic',msh=msh):
    mp,bs = msh.boundaries(np.pi/2,np.linspace(0,2*np.pi,50))

    fig,axs = plt.subplots(n,2,figsize=(15,n*6))
    for ax in axs.ravel():
        ax.plot(mp[1],mp[2],ls='-.',color='k')
        ax.plot(bs[1],bs[2],ls='--',color='k')
        ax.set_aspect('equal')
 
        ax.axhline(0,ls='--',c='k',alpha=0.75)
        ax.axvline(0,ls='--',c='k',alpha=0.75)
        ax.set_xlim(-16,16)
        ax.set_ylim(-16,16)
    
    alphabet = list('abcdefghijklmnopqr')
    apair= alphabet[::2] 
    aimpair= alphabet[1::2] 
    
    for i in range(len(axs)):
        axs[i,0].set_ylabel(r" $Z_{PGSM}$",fontsize=13)        
        #fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[i,1])
        axs[i,0].text(-18, 16.5, f'{apair[i]}.', fontsize=15)
        axs[i,1].text(-18, 16.5, f'{aimpair[i]}.', fontsize=15)

    for ax in axs.T: 
        ax[-1].set_xlabel(r"$Y_{PGSM}$",fontsize=13)
    

    fig.tight_layout()

    return fig,axs 


def fig_nxm_plots(n=2,m=2,norm=Normalize(-60,60),cmap='seismic',msh=msh):
    mp,bs = msh.boundaries(np.pi/2,np.linspace(0,2*np.pi,50))

    fig,axs = plt.subplots(m,n,figsize=(n*7.,m*6+0.25))
    for ax in axs.ravel():
        ax.plot(mp[1],mp[2],ls='-.',color='k')
        ax.plot(bs[1],bs[2],ls='--',color='k')
        ax.set_aspect('equal')
 
        ax.axhline(0,ls='--',c='k',alpha=0.75)
        ax.axvline(0,ls='--',c='k',alpha=0.75)
        ax.set_xlim(-16,16)
        ax.set_ylim(-16,16)
    
    alphabet = list('abcdefghijklmnopqr')
    apair= alphabet[::2] 
    aimpair= alphabet[1::2] 
    if len(axs.shape)>1:
        for i in range(len(axs)):
            axs[i,0].set_ylabel(r" $Z_{PGSM}$",fontsize=13)        
            #fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),ax=axs[i,1])
            axs[i,0].text(-18, 16.5, f'{apair[i]}.', fontsize=15)
            axs[i,1].text(-18, 16.5, f'{aimpair[i]}.', fontsize=15)

        for ax in axs.T: 
            ax[-1].set_xlabel(r"$Y_{PGSM}$",fontsize=13)
    fig.suptitle('   ')

    fig.tight_layout()

    return fig,axs 


def filter_nan_gaussian_conserving2(arr, sigma):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = gf(
            loss, sigma=sigma, mode='constant', cval=1)

    gauss = arr / (1-loss)
    gauss[nan_msk] = 0
    gauss = gf(
            gauss, sigma=sigma, mode='constant', cval=0)
    gauss[nan_msk] = np.nan

    return gauss


def train_KNN(x,y,N=10000):
    model = KNeighborsRegressor(n_neighbors=N,weights='distance',n_jobs=1)
    model.fit(x, y)
    return model

def select_data_with_tilt(wanted_tilt, tilt, data, dtilt=2.5):
    '''
    data can be a list of dataset
    return subset of data and the tilt
    '''
    cond = (abs(tilt)<=abs(wanted_tilt)+dtilt) & (abs(tilt)>=abs(wanted_tilt)-dtilt)
    data = su.listify(data)+[tilt]
    data = su.select_data_with_condition(data,cond)
    return data
    
def magnetosphere_symmetry_in_Z_for_vectorial_qty(wanted_tilt, tilt, pos, data ):
    if np.sign(wanted_tilt)!=0:
        if np.sign(wanted_tilt)<0:
            cond_sym = tilt>0
        elif np.sign(wanted_tilt)>0:
            cond_sym = tilt<0
        pos[:,2][cond_sym] = -pos[:,2][cond_sym] 
        data[:,:2][cond_sym] = -data[:,:2][cond_sym]
        tilt[cond_sym]=-tilt[cond_sym]
    else :
        possym= pos.copy()
        datasym= data.copy()
        possym[:,2] = -possym[:,2] 
        datasym[:,:2] = -datasym[:,:2]
        pos = np.concatenate((pos,possym),axis=0)
        data = np.concatenate((data,datasym),axis=0)
        tilt = np.concatenate((tilt,-tilt),axis=0)
        del possym,datasym
    return pos,data,tilt

def magnetosphere_symmetry_in_Z_for_scalar_qty(wanted_tilt, tilt, pos, data ):
    if np.sign(wanted_tilt)!=0:
        if np.sign(wanted_tilt)<0:
            cond_sym = tilt>0
        elif np.sign(wanted_tilt)>0:
            cond_sym = tilt<0
        pos[:,2][cond_sym] = -pos[:,2][cond_sym] 
        tilt[cond_sym]=-tilt[cond_sym]
    else :
        possym= pos.copy()
        possym[:,2] = -possym[:,2] 
        pos = np.concatenate((pos,possym),axis=0)
        data = np.concatenate((data,data),axis=0)
        tilt = np.concatenate((tilt,-tilt),axis=0)
        del possym
    return pos,data,tilt

def magnetosphere_symmetry_in_Y_for_vectorial_qty( tilt, pos, data ):
    possym= pos.copy()
    datasym= data.copy()
    possym[:,1] = -possym[:,1] 
    datasym[:,1] = -datasym[:,1]
    pos = np.concatenate((pos,possym),axis=0)
    data = np.concatenate((data,datasym),axis=0)
    tilt = np.concatenate((tilt,tilt),axis=0)
    del  possym,datasym
    return pos,data,tilt

def magnetosphere_symmetry_in_Y_for_scalar_qty( tilt, pos, data ):
    possym= pos.copy()
    possym[:,1] = -possym[:,1] 
    pos = np.concatenate((pos,possym),axis=0)
    data = np.concatenate((data,data),axis=0)
    tilt = np.concatenate((tilt,tilt),axis=0)
    del  possym
    return pos,data,tilt

    
def select_and_symmetry_with_tilt(wanted_tilt, tilt, pos, data, dtilt=2.5, z_sym=True, y_sym =True, verbose =True ):
    pos,data,tilt = select_data_with_tilt(wanted_tilt, tilt, [pos,data], dtilt=dtilt)
    if verbose :
        print('size before symmetry : ', len(pos))
        
    if z_sym :
        if len(data.shape)>1:
            pos,data,tilt = magnetosphere_symmetry_in_Z_for_vectorial_qty(wanted_tilt, tilt, pos, data )
        else :
            pos,data,tilt = magnetosphere_symmetry_in_Z_for_scalar_qty(wanted_tilt, tilt, pos, data )
        
    if y_sym :
        if len(data.shape)>1:
            pos,data,tilt = magnetosphere_symmetry_in_Y_for_vectorial_qty( tilt, pos, data )
        else :
            pos,data,tilt = magnetosphere_symmetry_in_Y_for_scalar_qty( tilt, pos, data )
        
    if verbose :
        print("mean tilt : ", np.mean(tilt), '   size dataset :',len(pos))
    return pos,data,tilt


def magnetosheath_symmetry_in_Z_for_vectorial_qty(pos, data, omni):
    possym= pos.copy()
    datasym= data.copy()
    possym[:,2] = -possym[:,2] 
    datasym[:,2] = -datasym[:,2]
    pos = np.concatenate((pos,possym),axis=0)
    data = np.concatenate((data,datasym),axis=0)
    omni = pd.concat((omni,omni),axis=0)
    del possym,datasym
    return pos,data,omni

def magnetosheath_symmetry_in_Z_for_scalar_qty(pos, data ):
    possym= pos.copy()
    possym[:,2] = -possym[:,2] 
    pos = np.concatenate((pos,possym),axis=0)
    data = np.concatenate((data,data),axis=0) 
    omni = pd.concat((omni,omni),axis=0)
    del possym
    return pos,data,omni
    
def associate_SW_Safrankova(X_sat, omni, BS_standoff, dtm=0,sampling_time='5S',vx_median =-406.2,mean_rol=False):
    if dtm != 0:
        #vxmean = abs(omni.Vx.rolling(dt,min_periods=1).mean())
        vxmean = abs(omni.Vx.rolling(dtm,center=True).mean())
        #vxmean = abs(omni.Vx.rolling(int((2*dtm+1)*timedelta(minutes=1)/(omni.index[-1]-omni.index[-2])),center=True,min_periods=1).mean())
    else:
        vxmean = abs(omni.Vx)
        
    if (mean_rol ==True) & (dtm != 0):
        omni_mean = abs(omni.rolling(dtm,center=True).mean())
    else :   
        omni_mean =omni
    
    BS_x0 = BS_standoff[BS_standoff.index.isin(X_sat.index)]
    BS_x0 = BS_x0.fillna(13.45)
    lag = np.array(np.round((BS_x0.values-X_sat.values)*6371/vx_median),dtype='timedelta64[s]')
    time = (X_sat.index-lag).round(sampling_time)
    vx = pd.Series(name='Vx',dtype=float)
    vx  = vx.append(vxmean.loc[time],ignore_index=True).fillna(abs(vx_median)).values
    lag = np.array(np.round((BS_x0.values-X_sat.values)*6371/vx),dtype='timedelta64[s]')
    print((X_sat.index-lag),(X_sat.index-lag).round(sampling_time))
    time = (X_sat.index-lag).round(sampling_time)
    OMNI = pd.DataFrame(columns=omni.columns)
    OMNI = OMNI.append(omni_mean.loc[time], ignore_index=True)
    OMNI.index = X_sat.index
    return OMNI.dropna(),lag

def make_vectorial_qty_msp(knn_function,coord, interpBmsp, regular_coord=None,sigma=20,rm_normal=True):
    L=[]
    L.append(np.array(coord).T)
    L.append(interpBmsp)

    #L remplace [np.array(coord).T,interpBmsp]
    qtyx,qtyy,qtyz = compute_knn(knn_function,L).T
    if regular_coord is not None:
        qtyx,qtyy,qtyz = smap.interpolate_on_regular_grid(coord[1],coord[2],[qtyx,qtyy,qtyz],regular_coord[1],regular_coord[2])

    else:
        regular_coord[0],regular_coord[1],regular_coord[2] = coord[0],coord[1],coord[2]
    if sigma !=0:
        qtyx,qtyy,qtyz = smap.make_gaussian_filter([qtyx,qtyy,qtyz],(sigma,sigma))
    if rm_normal :
        qtyx,qtyy,qtyz = smap.remove_normal_to_shue98(regular_coord[0],regular_coord[1],regular_coord[2],qtyx,qtyy,qtyz)
    return qtyx,qtyy,qtyz

def make_scalar_qty_msp(knn_function,coord, interpNpmsp, regular_coord=None,sigma=20):
    L=[]
    L.append(np.array(coord).T)
    L.append(interpNpmsp)
    
    qty = compute_knn(knn_function,L).T
    if regular_coord is not None:
        qty = smap.interpolate_on_regular_grid(coord[1],coord[2],[qty],regular_coord[1],regular_coord[2])[0]
    else:
        regular_coord[0],regular_coord[1],regular_coord[2] = coord[0],coord[1],coord[2]
    if sigma !=0:
        qty= smap.make_gaussian_filter([qty],(sigma,sigma))[0]
    return qty
        
#Le code à optimiser
def transform_vectorial_qty_msh_swi(qty0,coord,regular_coord=None, negative_bximf=False ,new_clock=None, old_clock = np.radians(90),sigma=20,rm_normal=True):   
    qtyx,qtyy,qtyz = qty0[0].copy(),qty0[1].copy(),qty0[2].copy()
    if negative_bximf:
        qtyx,qtyy,qtyz  = smap.swi_to_negative_bximf(coord[1],coord[2],qtyx,qtyy,qtyz)
      #  print('checked if negative'+datetime.datetime.now())
    if new_clock is not None:
        xr,yr,zr,qtyx,qtyy,qtyz = smap.rotates_clock_angle(coord[0],coord[1],coord[2],qtyx,qtyy,qtyz,new_clock,old_clock)
      #  print('changed rotation'+datetime.datetime.now())
    else :
        xr,yr,zr = coord[0],coord[1],coord[2]
    if regular_coord is not None:
        qtyx,qtyy,qtyz = smap.interpolate_on_regular_grid(yr,zr,[qtyx,qtyy,qtyz],regular_coord[1],regular_coord[2])
       # print('checked regular coordonates'+datetime.datetime.now())
    else:
        regular_coord[0],regular_coord[1],regular_coord[2] = xr,yr,zr
       # print('checked regular coordonates'+datetime.datetime.now())
    if sigma !=0:
        qtyx,qtyy,qtyz = smap.make_gaussian_filter([qtyx,qtyy,qtyz],(sigma,sigma))
        #print('Gaussian filter for sigma'+datetime.datetime.now())
    if rm_normal :
        qtyx,qtyy,qtyz = smap.remove_normal_to_shue98(regular_coord[0],regular_coord[1],regular_coord[2],qtyx,qtyy,qtyz)
       # print('rmnormal?'+datetime.datetime.now())
    return qtyx,qtyy,qtyz
    
def transform_scalar_qty_msh_swi(qty0,coord,regular_coord=None, new_clock=None, old_clock = np.radians(90),sigma=20):
    if new_clock is not None:
        xr,yr,zr = smap.rotates_phi_angle(coord[0],coord[1],coord[2],new_clock - old_clock)
        #print('rotates but for scalar qty'+datetime.datetime.now())
    else:
        xr,yr,zr= coord
        #print('rotates but for scalar qty'+datetime.datetime.now())
    if regular_coord is not None:
        qty = smap.interpolate_on_regular_grid(yr,zr,[qty0],regular_coord[1],regular_coord[2])[0]
        #print('checked coordonates'+datetime.datetime.now())
    else :
        qty = smap.interpolate_on_regular_grid(yr,zr,[qty0],coord[1],coord[2])[0]
        #print('checked coordonates'+datetime.datetime.now())
    if sigma!=0:
        qty = smap.make_gaussian_filter([qty],(sigma,sigma))[0]
        #print('Gaussian filter for sigma'+datetime.datetime.now())
    return qty

def magnetic_field_from_data(xmp,ymp,zmp,interp,**kwargs):
    if kwargs.get('knn_function', None) is not None:
        bx, by, bz = compute_knn(kwargs['knn_function'],[np.array([xmp,ymp,zmp]).T, interp]).T
    elif kwargs.get('B', None) is not None:
        bx, by, bz  = kwargs['B']
    else :
        raise ValueError('kwargs should contain a knn_function or B argument')
        
    if kwargs.get( 'new_clock' , None) is not None :
        xmp_new, ymp_new, zmp_new, bx_new, by_new, bz_new = rotates_clock_angle(xmp,ymp,zmp, bx, by, bz,kwargs['new_clock'], kwargs.get( 'old_clock' , np.pi/2))
    else :
        xmp_new, ymp_new, zmp_new, bx_new, by_new, bz_new = xmp, ymp, zmp, bx, by, bz
    
    
    if kwargs.get( 'sigma' , None) is not None :
        if np.sum([assert_regularity_grid(g) for g in [xmp_new, ymp_new, zmp_new]])<2:
            ValueError('The grid should be regular')    
        bx_new, by_new, bz_new  = make_gaussian_filter([bx_new, by_new, bz_new],kwargs['sigma'])
    
    if kwargs.get( 'remove_bn' , True):
        bx_new, by_new, bz_new = remove_normal_to_shue98(xmp_new, ymp_new, zmp_new, bx_new, by_new, bz_new)
    
    return xmp_new, ymp_new, zmp_new, bx_new, by_new, bz_new


def make_current_density(xx,yy,zz,bxmsp,bymsp,bzmsp,bxmsh,bymsh,bzmsh,bimf_norm=5, dmp = 800/6400):
    bmsp_norm = sm.norm(bxmsp,bymsp,bzmsp)
    lx,ly,lz = bxmsp/bmsp_norm,bymsp/bmsp_norm,bzmsp/bmsp_norm
    Blmsh = bxmsh*lx + bymsh*ly + bzmsh*lz
    Blmsp = bxmsp*lx + bymsp*ly + bzmsp*lz
    th,ph =scc.cartesian_to_spherical(xx,yy,zz)[1:]
    nx,ny,nz = smp.mp_shue1998_normal(th,ph)
    mx,my,mz = np.cross(np.asarray([nx,ny,nz]).T,np.asarray([lx,ly,lz]).T).T
    Bmmsh = bxmsh*mx + bymsh*my + bzmsh*mz
    Bmmsp = bxmsp*mx + bymsp*my + bzmsp*mz
    jl=-(Bmmsh*bimf_norm-Bmmsp)*1e-9/(cst.mu_0*dmp*6400*1e3)
    jm=(Blmsh*bimf_norm-Blmsp)*1e-9/(cst.mu_0*dmp*6400*1e3)
    jj= sm.norm(0,jl,jm)*1e9
    jx = (jm*mx+jl*lx)*1e9
    jy = (jm*my+jl*ly)*1e9
    jz = (jm*mz+jl*lz)*1e9
    return jx,jy,jz,jj

def make_RR(npmsp,npmsh,bmsp,bmsh,alpha,bimf_norm=5):   
    b1 = sm.norm(bmsp[0],bmsp[1],bmsp[2])*1e-9*np.sin(np.radians(alpha)/2)
    b2 = sm.norm(bmsh[0],bmsh[1],bmsh[2])*1e-9*bimf_norm*np.sin(np.radians(alpha)/2)
    np1 = npmsp
    np2 = npmsh
    R= 2*(0.1*(b2*b1)**(3/2))/(np.sqrt(cst.mu_0)*np.sqrt(b2*np1+b1*np2)*np.sqrt(b1+b2)) *1e3
    return R

def make_RR_with_shear_flow(npmsp,npmsh,bmsp,bmsh,vmsp,vmsh,alpha,bimf_norm=5,vsw_norm=400):   
    b1 = sm.norm(bmsp[0],bmsp[1],bmsp[2])*1e-9*np.sin(np.radians(alpha)/2)
    b2 = sm.norm(bmsh[0],bmsh[1],bmsh[2])*1e-9*bimf_norm*np.sin(np.radians(alpha)/2)
    v1 = sm.norm(vmsp[0],vmsp[1],vmsp[2])*1e3*np.sin(np.radians(alpha)/2)
    v2 = sm.norm(vmsh[0],vmsh[1],vmsh[2])*1e3*vsw_norm*np.sin(np.radians(alpha)/2)
    np1 = npmsp
    np2 = npmsh
    R= 2*(0.1*(b2*b1)**(3/2))/(np.sqrt(cst.mu_0)*np.sqrt(b2*np1+b1*np2)*np.sqrt(b1+b2)) *1e3
    ca= np.sqrt((b1*b2*(b1+b2))/(cst.mu_0*(b2*np1+b1*np2)))
    vshear = (v2-v1)/2
    A = (vshear/ca)**2
    B = 4*np1*b2*np2*b1
    C = (b2*np1+b1*np2)**2
    Rv= R*(1-A*B/C)
    return Rv

def compute_knn(knn_function,var):
    #print('L type dans computeknn', type(var))
    #print('Type de interpBmsp dans computeknn', type(var[1]))
    with Pool(85) as pool :
        qty= pool.map(partial(knn_function,var[1]),var[0])
    pool.close()
    return np.asarray(qty)

#var=[p0,interpxxx]
def f_bmsh_pool(interpBmsh, p0):
    b = interpBmsh.predict(p0)
    return b

def f_bmsp_pool(interpBmsp, p0):

    b = interpBmsp.predict(p0)
    return b

def f_nmsh_pool(interpNpmsh, p0):
    b = interpNpmsh.predict(p0)
    return b

def f_nmsp_pool(interpNpmsp, p0):
    b = interpNpmsp.predict(p0)
    return b

def f_vmsh_pool(p0,interpVmsh):
    b = interpVmsh.predict(p0)
    return b

def f_vmsp_pool(p0,interpVmsp):
    b = interpVmsp.predict(p0)
    return b

def read_gsm_files(omni_gsm, data_gsm, pos_gsm):
    
    tilt = np.degrees(data_gsm.tilt)
    
    Bmsp =data_gsm[['Bx','By','Bz']].values
    Npmsp = data_gsm.Np.values
    Vmsp = data_gsm[['Vx','Vy','Vz']].values
    
    pmsp = pos_gsm[['X','Y','Z']].values
    del data_gsm
    del pos_gsm
    
    return(tilt, Bmsp, Npmsp, Vmsp, pmsp)
    
def data_according_to_cone(coa, data_coa, data_swi, pos_swi):

    Bmsh =data_swi[['Bx','By','Bz']].copy()
    Bmsh.Bx = Bmsh.Bx/data_coa.B
    Bmsh.By = Bmsh.By/data_coa.B
    Bmsh.Bz = Bmsh.Bz/data_coa.B
    Bmsh = Bmsh.values

    Vmsh =data_swi[['Vx','Vy','Vz']].copy()
    Vmsh.Vx = Vmsh.Vx/data_coa.V
    Vmsh.Vy = Vmsh.Vy/data_coa.V
    Vmsh.Vz = Vmsh.Vz/data_coa.V
    Vmsh = Vmsh.values

    Npmsh = (data_swi.Np/data_coa.Np).values


    del data_swi
    pmsh = pos_swi[['X','Y','Z']].values
    del pos_swi
    
    cond = (np.degrees(abs(data_coa.COA))>=(abs(coa)-5)) & (np.degrees(abs(data_coa.COA))<=(abs(coa)+5))

    p,b,n,v,o = su.select_data_with_condition([pmsh,Bmsh,Npmsh,Vmsh,data_coa],cond)

    pp,bb,oo = magnetosheath_symmetry_in_Z_for_vectorial_qty(p,b,o)
    print(len(pp),np.degrees(abs(oo.COA).mean()))
    
    return pp, bb, p, n

def creation_interpolations(cone, tilt, datagsm, dataswi):
    
    omni_gsm, data_gsm, pos_gsm=datagsm[0],datagsm[1],datagsm[2]
    omni_swi, data_swi, pos_swi=dataswi[0],dataswi[1],dataswi[2]
    (tilt_read, Bmsp, Npmsp, Vmsp, pmsp) = read_gsm_files(omni_gsm, data_gsm, pos_gsm)
    
    pt,bt,selected_tilt = select_and_symmetry_with_tilt(tilt, tilt_read, pmsp,Bmsp, dtilt=2.5)
    pt2,vt,npt = select_data_with_tilt(tilt, tilt_read, [pmsp,Vmsp,Npmsp], dtilt=2.5)[:3]
    
    interpBmsp = train_KNN(pt,bt,N=10000)
    interpNpmsp = train_KNN(pt2,npt,N=10000)
    interpVmsp = train_KNN(pt2,vt,N=10000)
        
    pp,bb,p,n= data_according_to_cone(cone,omni_swi,data_swi, pos_swi)
    
    interpBmsh = train_KNN(pp,bb,N=10000) #nombre de points N sera rentré par utilisateur?
    interpNpmsh = train_KNN(p,n,N=10000)
    
    return interpBmsh, interpNpmsh, interpBmsp, interpNpmsp

def import_files():
    #Adresse à changer?
    omni_swi= pd.read_pickle('/DATA/michotte/MSH_data/5S_V2/omni_swi.pkl')
    data_swi= pd.read_pickle('/DATA/michotte/MSH_data/5S_V2/data_swi.pkl')
    pos_swi= pd.read_pickle('/DATA/michotte/MSH_data/5S_V2/pos_swi.pkl')

    omni_gsm= pd.read_pickle('/DATA/michotte/MSP_data/5S_V1/omni_gsm.pkl')
    data_gsm= pd.read_pickle('/DATA/michotte/MSP_data/5S_V1/data_gsm.pkl')
    pos_gsm= pd.read_pickle('/DATA/michotte/MSP_data/5S_V1/pos_gsm.pkl')
    print("imported files")
    return(omni_swi,data_swi,pos_swi,omni_gsm,data_gsm,pos_gsm)

def enregistrement_pkl(cone,tilt,Xmp,Ymp,Zmp,xx,yy,zz,npmsh,npmsp,bxmsp,bymsp,bzmsp,bxmsh,bymsh,bzmsh):
#Adresse a changer?
    pd.to_pickle({'x':Xmp,'y':Ymp,'z':Zmp,'Bx':bxmsh,'By':bymsh,'Bz':bzmsh},f'grid_b_msh_{cone}.pkl')
    pd.to_pickle({'x':Xmp,'y':Ymp,'z':Zmp,'Bx':bxmsp,'By':bymsp,'Bz':bzmsp},f'grid_b_msp_{cone}_{tilt}.pkl')
    
    pd.to_pickle({'x':Xmp,'y':Ymp,'z':Zmp,'Np':npmsh},f'grid_np_msh_{cone}.pkl')
    pd.to_pickle({'x':xx,'y':yy,'z':zz,'Np':npmsp},f'grid_np_msp_{cone}_{tilt}.pkl')
    print('grilles enregistrées')
    

    
###################################################################################################################################################################################      
    
def grid_creation(cone, tilt):
    
    #read files
    omni_swi,data_swi,pos_swi,omni_gsm,data_gsm,pos_gsm=import_files()
    ([xx,yy,zz],[Xmp, Ymp, Zmp]) = smapplt.make_reg_grid()
    
    print('creation données interpolation')
    Npmsh = (data_swi.Np/omni_swi.Np).values
    interpBmsh, interpNpmsh, interpBmsp, interpNpmsp=creation_interpolations(cone, tilt, [omni_gsm, data_gsm, pos_gsm], [omni_swi, data_swi, pos_swi])
    
    print('creation npmsp')
    npmsp = make_scalar_qty_msp(f_nmsp_pool,[Xmp,Ymp,Zmp],interpNpmsp, regular_coord=[xx,yy,zz],sigma=20) *1e6*cst.m_p
    print('création bmsp')
    bxmsp,bymsp,bzmsp = make_vectorial_qty_msp(f_bmsp_pool,[Xmp,Ymp,Zmp],interpBmsp, regular_coord=[xx,yy,zz],sigma=20,rm_normal=True)
    bmsp = [bxmsp,bymsp,bzmsp]
    print('création npmsh')
    npmsh0 = compute_knn(f_nmsh_pool,[np.asarray([Xmp,Ymp,Zmp]).T, interpNpmsh]).T*1e6*cst.m_p*Npmsh.mean()#o.Np.mean()####compute_knn and magnetic field from data a adapter
    print('création bmsh')
    bxmsh,bymsh,bzmsh=compute_knn(f_bmsh_pool,[np.array([Xmp,Ymp,Zmp]).T,interpBmsh]).T
    
    
    
    print('enregistrement des pkl')
    enregistrement_pkl(cone,tilt,Xmp,Ymp,Zmp,xx,yy,zz,npmsh0,npmsp,bxmsp,bymsp,bzmsp,bxmsh,bymsh,bzmsh)
    