import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool 
from scipy.integrate import solve_ivp
from skimage.feature import hessian_matrix,hessian_matrix_eigvals
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter as gf
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import fsolve
from math import radians as rad
from math import degrees as deg

sys.path.append('Notebooks/space')
sys.path.append('.')
sys.path.insert(0,'../py_tsyganenko/build/')


from space.models import planetary as smp
from space import smath as sm
from space.coordinates import coordinates as scc
from space import utils as su
from skimage.feature import peak_local_max


from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
msh = smp.Magnetosheath(magnetopause='mp_shue1998', bow_shock ='bs_jelinek2012')




from utilities import pandas_fill, reshape_to_2Darrays, reshape_to_original_shape,filter_nan_gaussian_conserving2,assert_regularity_grid
import py_tsyganenko.Geopack as gp
import py_tsyganenko.Models as tm
import datetime

from space.models import planetary as smp
from space import smath as sm
from space.coordinates import coordinates as scc
from space import utils as su
#from skimage.feature import peak_local_max



def find_max_point(x,y,qty,n=3,rlim_max=9,min_distance=5,indexing='xy',verbose=True):
    coord = peak_local_max(qty, min_distance=min_distance)
    xm = np.asarray([x[c[0],c[1]] for c in coord])
    ym = np.asarray([y[c[0],c[1]] for c in coord])
    qm = np.asarray([qty[c[0],c[1]] for c in coord])
    r = sm.norm(xm,ym,0)
    xm ,ym ,qm = xm[r<=rlim_max], ym[r<=rlim_max],qm[r<=rlim_max]
    #if len(xm)>=1:
    #    return xm[qm.argmax()],ym[qm.argmax()]
    #else :
    #    return np.nan,np.nan
    return xm[qm.argmax()],ym[qm.argmax()]


def date_for_recalc(date):
    '''
    Input :
        date : str
             format : 'DD-MM-YYYY hh:mm:ss'
        
    Output :
        year : int 
        doy : int
        hour : int
        mins : int
        secs : int
    '''

    str_date=su.listify(date)
    date = pd.DatetimeIndex(str_date,dayfirst=True)
    year = pd.DatetimeIndex(date).year[0]
    doy  = date.day_of_year[0]
    hour = date.hour[0]
    mins = date.minute[0]
    secs = date.second[0]
    return year,doy,hour,mins,secs

def tilt_geopack(date,vx=-400,vy=0,vz=0):
    year,doy,hour,mins,secs = date_for_recalc(date)
    gp.recalc(year,doy,hour,mins,secs,vx,vy,vz) 
    return gp.GEOPACK1.PSI

def gsm_to_gsw(date,xgsm,ygsm,zgsm,vx,vy,vz,vcoord='gsm'):
    '''
    if vx,vy, and vz are in gse coordinates, vcoord must be change to 'gse'
    '''
    if vy==0  and vz==0 :
        xgsw,ygsw,zgsw = xgsm,ygsm,zgsm
    else: 

        xgse,ygse,zgse = gsm_to_gse(date,xgsm,ygsm,zgsm)
        xgsw,ygsw,zgsw = gse_to_gsw(date,xgse,ygse,zgse,vx,vy,vz,vcoord=vcoord)
        
    return xgsw,ygsw,zgsw


def gsw_to_gsm(date,xgsw,ygsw,zgsw,vx,vy,vz,vcoord='gsm'):
    '''
    if vx,vy, and vz are in gse coordinates, vcoord must be change to 'gse'
    '''
    if vy==0  and vz==0 :
         xgsm,ygsm,zgsm =xgsw,ygsw,zgsw
    else: 
        xgse,ygse,zgse = gsw_to_gse(date,xgsw,ygsw,zgsw,vx,vy,vz,vcoord=vcoord)
        xgsm,ygsm,zgsm = gse_to_gsm(date,xgse,ygse,zgse)
    return xgsm,ygsm,zgsm

def gse_to_gsw(date,xgse,ygse,zgse,vx,vy,vz,vcoord='gsm'):
    '''
    if vx,vy, and vz are in gse coordinates, vcoord must be change to 'gse'
    '''
 
    vxgse,vygse,vzgse = velocity_to_gse(date,vx,vy,vz,vcoord=vcoord)
    pos_gse,old_shape = reshape_to_2Darrays([xgse,ygse,zgse])
    year,doy,hour,mins,secs = date_for_recalc(date)
    gp.recalc(year,doy,hour,mins,secs,vxgse,vygse,vzgse) 
    pos_gsw = gp.gse_to_gsw(pos_gse) 
    xgsw,ygsw,zgsw = reshape_to_original_shape(pos_gsw, old_shape)
    return xgsw,ygsw,zgsw

def gsw_to_gse(date,xgsw,ygsw,zgsw,vx,vy,vz,vcoord='gsm'):
    '''
    if vx,vy, and vz are in gse coordinates, vcoord must be change to 'gse'
    '''
    vxgse,vygse,vzgse = velocity_to_gse(date,vx,vy,vz,vcoord=vcoord)
    pos_gsw,old_shape = reshape_to_2Darrays([xgsw,ygsw,zgsw])
    year,doy,hour,mins,secs = date_for_recalc(date)
    gp.recalc(year,doy,hour,mins,secs,vxgse,vygse,vzgse) 
    pos_gse = gp.gsw_to_gse(pos_gsw)
    xgse,ygse,zgse = reshape_to_original_shape(pos_gse, old_shape)
    return xgse,ygse,zgse

def gsm_to_gse(date,xgsm,ygsm,zgsm):
    pos_gsm,old_shape = reshape_to_2Darrays([xgsm,ygsm,zgsm])
    year,doy,hour,mins,secs = date_for_recalc(date)
    gp.recalc(year,doy,hour,mins,secs,-400,0,0) 
    pos_gse = gp.gsw_to_gse(pos_gsm)
    xgse,ygse,zgse = reshape_to_original_shape(pos_gse, old_shape)
    return xgse,ygse,zgse
    
def gse_to_gsm(date,xgse,ygse,zgse):
    pos_gse,old_shape = reshape_to_2Darrays([xgse,ygse,zgse])
    year,doy,hour,mins,secs = date_for_recalc(date)
    gp.recalc(year,doy,hour,mins,secs,-400,0,0) 
    pos_gse = gp.gse_to_gsw(pos_gse)
    xgsm,ygsm,zgsm = reshape_to_original_shape(pos_gse, old_shape)
    return xgsm,ygsm,zgsm

def velocity_to_gse(date,vx,vy,vz,vcoord='gsm'):
    '''
    will change the velocity from gsm to gse coordinates
    '''
    if vcoord=='gsm' :
        vxgse,vygse,vzgse = gsm_to_gse(date,vx,vy,vz)
        vxgse,vygse,vzgse = vxgse[0],vygse[0],vzgse[0]
    elif vcoord=='gse':
        vxgse,vygse,vzgse = vx,vy,vz
    else:
        raise ValueError("Velocity coordinates must be in GSM (vcoord='gsm') or GSE (vcoord='gse')")
    return vxgse,vygse,vzgse 

def coord_sys_to_gsw(date,x,y,z,vx,vy,vz,xcoord='gsm',vcoord='gsm'):
    if xcoord=='gsm':
        xgsw,ygsw,zgsw = gsm_to_gsw(date,x,y,z,vx,vy,vz,vcoord=vcoord)
    elif xcoord=='gse':
        xgsw,ygsw,zgsw = gse_to_gsw(date,x,y,z,vx,vy,vz,vcoord=vcoord)
    elif xcoord=='gsw':
        xgsw,ygsw,zgsw = x,y,z
    else:
        raise ValueError("Position coordinates must be in GSM (xcoord='gsm') or GSE (xcoord='gse') or GSW (xcoord='gsw')")
    return xgsw,ygsw,zgsw
    
def gsw_to_coord_sys(date,xgsw,ygsw,zgsw,vx,vy,vz,xcoord='gsm',vcoord='gsm'):
    if xcoord=='gsm':
        x,y,z = gsw_to_gsm(date,xgsw,ygsw,zgsw,vx,vy,vz,vcoord=vcoord)
    elif xcoord=='gse':
        x,y,z = gsw_to_gse(date,xgsw,ygsw,zgsw,vx,vy,vz,vcoord=vcoord)
    elif xcoord=='gsw':
        x,y,z =xgsw,ygsw,zgsw 
    else:
        raise ValueError("Position coordinates must be in GSM (xcoord='gsm') or GSE (xcoord='gse') or GSW (xcoord='gsw')")
    return x,y,z

def coord_sys_to_gsm(date,x,y,z,vx,vy,vz,xcoord='gsm',vcoord='gsm'):
    if xcoord=='gsw':
        xgsm,ygsm,zgsm = gsw_to_gsm(date,x,y,z,vx,vy,vz,vcoord=vcoord)
    elif xcoord=='gse':
        xgsm,ygsm,zgsm = gse_to_gsm(date,x,y,z)
    elif xcoord=='gsm':
        xgsm,ygsm,zgsm = x,y,z
    else:
        raise ValueError("Position coordinates must be in GSM (xcoord='gsm') or GSE (xcoord='gse') or GSW (xcoord='gsw')")
    return xgsm,ygsm,zgsm

def gsm_to_coord_sys(date,xgsm,ygsm,zgsm,vx,vy,vz,xcoord='gsm',vcoord='gsm'):
    if xcoord=='gsw':
        x,y,z = gsm_to_gsw(date,xgsm,ygsm,zgsm,vx,vy,vz,vcoord=vcoord)
    elif xcoord=='gse':
        x,y,z = gsm_to_gse(date,xgsm,ygsm,zgsm)
    elif xcoord=='gsm':
        x,y,z = xgsm,ygsm,zgsm
    else :
        raise ValueError("Position coordinates must be in GSM (xcoord='gsm') or GSE (xcoord='gse') or GSW (xcoord='gsw')")
    return x,y,z


def mp_sibeck_for_t96(theta,phi,**kwargs):
    Pd = kwargs.get('Pd', 2.056)
    a0 = 0.14
    b0 = 18.2
    c0 = -217.2
    p0 = 2.04
    
    a = a0 * np.cos(theta) ** 2 + np.sin(theta) ** 2
    b = b0 * np.cos(theta)
    c = c0 
    r = sm.resolve_poly2_real_roots(a, b, c)[0]
    r = r*(p0/Pd)**0.158
    return scc.choice_coordinate_system(r, theta, phi, **kwargs)

def mp_sibeck1991_for_t96_tangents(theta, phi, **kwargs):
    theta = su.listify(theta)
    phi = su.listify(phi)
    Pd = kwargs.get("Pd", 2.056)
    
    a0 = 0.14
    b0 = 18.2
    c0 = -217.2
    p0 = 2.04
    #p0 = 2.0
   

    a = a0 * np.cos(theta) ** 2 + np.sin(theta) ** 2
    dadt = 2 * np.cos(theta) * np.sin(theta) * (1 - a0)

    b = b0 * np.cos(theta) * (p0 / Pd) ** (1 / 6)
    dbdt = -b0 * np.sin(theta) * (p0 / Pd) ** (1 / 6)

    c = c0 * (p0 / Pd) ** (1 / 3)
    dcdt = 0

    delta = b ** 2 - 4 * a * c
    ddeltadt = 2 * b * dbdt - 4 * dadt * c

    u = -b + np.sqrt(delta)
    dudt = -dbdt + ddeltadt / (2 * np.sqrt(delta))

    v = 2 * a
    dvdt = 2 * dadt

    r = sm.resolve_poly2_real_roots(a, b, c)[0]
    drdt = (dudt * v - dvdt * u) / v ** 2
    drdp = 0

    return smp.derivative_spherical_to_cartesian(r, theta, phi, drdt, drdp)


def mp_sibeck1991_for_t96_normal(theta, phi, **kwargs):
    tth, tph = mp_sibeck1991_for_t96_tangents(theta, phi, **kwargs)
    return smp.find_normal_from_tangents(tth, tph)

def find_crossing_normal_mp_sibeck_kf94(xmp,ymp,nx,ny,x0,phi):
    nx,ny = su.listify(nx),su.listify(ny)
    k = np.zeros_like(nx)
    k[ny!=0] = (nx[ny!=0]/ny[ny!=0])
    a = 1/(2*x0)
    b = k*np.sin(phi)
    c = xmp-x0-k*ymp
    ryz = sm.resolve_poly2_real_roots(a,b,c)[0]
    
    x = x0-ryz**2/(2*x0)
    y = ryz*np.sin(phi)
    z = ryz*np.cos(phi)
    return x,y,z

def magnetic_field_igrf(date,x,y,z,vx,vy,vz,xcoord='gsm',vcoord='gsm'):
    '''
    if x, y, and z are in gse or gsw coordinates, xcoord must be change to 'gse' or 'gsw'.
    if vx, vy, and vz are in gse coordinates, vcoord must be change to 'gse'.
    
    Output : bx,by,bz igrf in the same coordinate system than the positions (xcoord)
    '''
    vxgse,vygse,vzgse = velocity_to_gse(date,vx,vy,vz,vcoord=vcoord) 
    xgsw,ygsw,zgsw = coord_sys_to_gsw(date,x,y,z,vxgse,vygse,vzgse,xcoord=xcoord,vcoord='gse')
    pos_gsw, old_shape = reshape_to_2Darrays([xgsw,ygsw,zgsw])
    year,doy,hour,mins,secs = date_for_recalc(date)
    gp.recalc(year,doy,hour,mins,secs, vxgse,vygse,vzgse)
    bgsw =  gp.igrf_gsw(pos_gsw)
    bxgsw,bygsw,bzgsw = reshape_to_original_shape(bgsw, old_shape)
    bx,by,bz = gsw_to_coord_sys(date,bxgsw,bygsw,bzgsw,vxgse,vygse,vzgse,xcoord=xcoord,vcoord='gse')
    return bx,by,bz

def magnetic_field_t96(date,x,y,z,vx,vy,vz,pdyn,dst,byimf,bzimf,ps,xcoord='gsm',vcoord='gsm'):
    '''
    if x, y, and z are in gse or gsw coordinates, xcoord must be change to 'gse' or 'gsw'.
    if vx, vy, and vz are in gse coordinates, vcoord must be change to 'gse'.
    
    Output : bx,by,bz from tsyganenko's 1996 model in the same coordinate system than the positions (xcoord)
    '''
    vxgse,vygse,vzgse = velocity_to_gse(date,vx,vy,vz,vcoord=vcoord) 
    xgsm,ygsm,zgsm = coord_sys_to_gsm(date,x,y,z,vxgse,vygse,vzgse,xcoord=xcoord,vcoord='gse')
    pos_gsm, old_shape = reshape_to_2Darrays([xgsm,ygsm,zgsm])
    year,doy,hour,mins,secs = date_for_recalc(date)
    gp.recalc(year,doy,hour,mins,secs, vxgse,vygse,vzgse)
    bgsm = tm.T96(pdyn,dst,byimf,bzimf,ps,pos_gsm)
    bxgsm,bygsm,bzgsm = reshape_to_original_shape(bgsm, old_shape)
    bx,by,bz = gsm_to_coord_sys(date,bxgsm,bygsm,bzgsm,vxgse,vygse,vzgse,xcoord=xcoord,vcoord='gse')
    return bx,by,bz
    

def kf94_field_mp_t96(theta,phi,pdyn, bximf,byimf,bzimf,bs_model = smp.bs_jelinek2012,coord=False):
    xmp,ymp,zmp = mp_sibeck_for_t96(theta,phi,Pd=pdyn)
    x0 = mp_sibeck_for_t96(0,0,Pd=pdyn)[0]
    x1 = bs_model(0,0)[0]
    nx,ny,nz = mp_sibeck1991_for_t96_normal(theta,phi)       
    xmsh,ymsh,zmsh = find_crossing_normal_mp_sibeck_kf94(xmp,ymp,nx,ny,x0,phi)
    bxmsh, bymsh, bzmsh = smp.KF1994(xmsh,ymsh,zmsh,x0,x1,bximf,byimf,bzimf)
    if coord:
        return bxmsh, bymsh, bzmsh,xmsh,ymsh,zmsh 
    else :
        return bxmsh, bymsh, bzmsh
    


def magnetospheric_field_t96(date,x,y,z,vx,vy,vz,pdyn,dst,byimf,bzimf,ps,xcoord='gsm',vcoord='gsm'):
    bxgsm0, bygsm0, bzgsm0 = magnetic_field_igrf(date,x,y,z,vx,vy,vz,xcoord=xcoord,vcoord=vcoord)
    bxgsm1, bygsm1, bzgsm1 =  magnetic_field_t96(date,x,y,z,vx,vy,vz,pdyn,dst,byimf,bzimf,ps,xcoord=xcoord,vcoord=vcoord)
    bxmsp, bymsp, bzmsp = add_igrf_tmodel(bxgsm0, bygsm0, bzgsm0,bxgsm1, bygsm1, bzgsm1)
    return bxmsp, bymsp, bzmsp 

def add_igrf_tmodel(bxgsm0, bygsm0, bzgsm0,bxgsm1, bygsm1, bzgsm1):
    return bxgsm0+bxgsm1, bygsm0+bygsm1, bzgsm0+bzgsm1

def shear_angle(Bxmsp, Bymsp, Bzmsp,Bxmsh, Bymsh, Bzmsh):
    dp = Bxmsh * Bxmsp + Bymsh * Bymsp + Bzmsh * Bzmsp
    Bmsh = sm.norm(Bxmsh, Bymsh, Bzmsh)
    Bmsp = sm.norm(Bxmsp,Bymsp,Bzmsp)
    shear = np.degrees(np.arccos(dp / (Bmsh * Bmsp)))
    return shear



def shear_map_t96_kf94(theta,phi,date,pdyn,vsw,bximf,byimf,bzimf,dst,ps,bs_model = smp.bs_jelinek2012):
    '''
    Vy,Vz, Byimf, and Bzimf are set to zero to calculate the magnetospheric field. 
    '''
    xmp,ymp,zmp = mp_sibeck_for_t96(theta, phi,Pd=pdyn)
    bxmsp, bymsp, bzmsp = magnetospheric_field_t96(date,xmp,ymp,zmp,-abs(vsw),0,0,pdyn,dst,0,0,ps,xcoord='gsm',vcoord='gse')
    bxmsh, bymsh, bzmsh = kf94_field_mp_t96(theta,phi,pdyn, bximf,byimf,bzimf,bs_model = bs_model)
    shear = shear_angle(bxmsp, bymsp, bzmsp, bxmsh, bymsh, bzmsh)
    return xmp,ymp,zmp,shear


def shear_map_from_t96_msh_data(theta,phi,date,pdyn,vsw,dst,ps,sigma=25,**kwargs):
    xmp,ymp,zmp = mp_sibeck_for_t96(theta,phi,Pd=pdyn)
    bxmsp, bymsp, bzmsp = magnetospheric_field_t96(date,xmp,ymp,zmp,-abs(vsw),0,0,pdyn,dst,0,0,ps,xcoord='gsm',vcoord='gse')
    x_msh,y_msh,z_msh, bxmsh, bymsh, bzmsh = magnetic_field_from_data(xmp,ymp,zmp,**kwargs)
    yy,zz = make_regular_grid(**kwargs)
    bxmsp, bymsp, bzmsp = interpolate_on_regular_grid(ymp,zmp,[bxmsp, bymsp, bzmsp],yy,zz,**kwargs)
    bxmsh, bymsh, bzmsh = make_gaussian_filter_for_non_regular_grid(y_msh,z_msh,[bxmsh, bymsh, bzmsh],yy,zz,sigma, **kwargs)
    shear = shear_angle(bxmsp, bymsp, bzmsp, bxmsh, bymsh, bzmsh)
    del xmp,ymp,zmp,x_msh,y_msh,z_msh,bxmsp, bymsp, bzmsp,bxmsh, bymsh, bzmsh
    return yy,zz,shear
    
def compute_knn(knn_function,p0):
    with Pool(85) as pool :
        qty = pool.map(knn_function,p0)
    pool.close()
    return np.asarray(qty)

def remove_normal_to_shue98(xmp,ymp,zmp,vx,vy,vz):
    theta,phi = scc.cartesian_to_spherical(xmp,ymp,zmp)[1:]
    nx,ny,nz = smp.mp_shue1998_normal(theta,phi)
    bn = nx*vx+ny*vy+nz*vz
    return vx-bn*nx,vy-bn*ny,vz-bn*nz
    
    
    
    

def magnetic_field_from_data(xmp,ymp,zmp,**kwargs):
    if kwargs.get('knn_function', None) is not None:
        bx, by, bz = compute_knn(kwargs['knn_function'],np.array([xmp,ymp,zmp]).T).T
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

            
def swi_to_negative_bximf(yy,zz,bx,by,bz):
    new_by = by.copy()
    new_bx,new_bz = interpolate_on_regular_grid(-yy,zz,[-bx,-bz],yy,zz)
    return new_bx,new_by,new_bz
    

def rotates_clock_angle(xmp,ymp,zmp,bx, by, bz, new_clock, old_clock):
    rotation_angle =new_clock - old_clock
    new_xmp,new_ymp,new_zmp = rotates_phi_angle(xmp,ymp,zmp,rotation_angle)
    bx_new, by_new, bz_new = rotates_phi_angle(bx, by, bz,rotation_angle)
    return new_xmp,new_ymp,new_zmp, bx_new, by_new, bz_new
        
def make_regular_grid(**kwargs):
    xlim = kwargs.get('xlim',(-20,20))
    ylim = kwargs.get('ylim',(-20,20))
    nb_pts = kwargs.get('nb_pts',401)
    x  = np.linspace(xlim[0],xlim[1],nb_pts)
    y  = np.linspace(ylim[0],ylim[1],nb_pts)
    xx,yy = np.meshgrid(x,y,indexing=kwargs.get('indexing','xy'))
    return xx,yy

def make_regular_interpolation(x, y, qty, new_x, new_y, **kwargs):
    method = kwargs.get('method','linear')
    
    qty_2d = reshape_to_2Darrays([qty])[0]
    xy =  reshape_to_2Darrays([x,y])[0]
    reg_qty = griddata(xy, qty_2d[:,0], (new_x, new_y), method=method) #linear nd interpolator
    #reg_qty = griddata(xy, qty_2d, (new_x, new_y))
    
    reg_qty = pandas_fill(reg_qty) #On tourne le B, et du coup les positions et le champ magnétique changent, yj et zj de dedépart deviennent yj', zj', mais il faut interpoler yi,zi, vers yiMSP et ziMSP avec Bij
    rmax = np.max(sm.norm(x,y,0)) 
    r = sm.norm(new_x, new_y,0)
    reg_qty[r>rmax] = np.nan
    return reg_qty

def interpolate_on_regular_grid(x,y,qties,xx,yy,**kwargs):
    
    if (not isinstance(qties,list)) and (not isinstance(qties,np.ndarray)) :
        qties = [qties]
    return [make_regular_interpolation(x, y, q, xx, yy, **kwargs)  for q in qties]

def make_gaussian_filter_for_non_regular_grid(x,y,qties,xx,yy,sigma, **kwargs):
    if (not isinstance(qties,list)) and (not isinstance(qties,np.ndarray)) :
        qties = [qties]
    qties  = interpolate_on_regular_grid(x, y, qties, xx, yy, **kwargs)  
    g_qties = make_gaussian_filter(qties,sigma)
    return g_qties

def make_gaussian_filter(qties,sigma):
    return [filter_nan_gaussian_conserving2(q,sigma) for q in qties]
    

def rotates_phi_angle(x,y,z,angle):
    r,th,ph = scc.cartesian_to_spherical(x,y,z)
    return scc.spherical_to_cartesian(r,th,ph+angle)


def gradient_2d_grid(yy,zz,qty,indexing='xy'):
    if indexing == 'ij' :
        yaxis=0
        zaxis=1
    else : 
        yaxis=1
        zaxis=0

    grad = np.gradient(qty)
    dy = np.gradient(yy)[yaxis]
    dz = np.gradient(zz)[zaxis]
    if (np.sum(np.diff(dy))>1e-10) or  (np.sum(np.diff(dz))>1e-10) :
        print('The grid is not regular')
    return grad[yaxis]/dy ,grad[zaxis]/dz

def find_potential_saddle_and_extremum_points(x,y,qty,n=3, threshold=1e-1,indexing='xy'):
    if n<2 :
        raise ValueError('n should be at superior or equal to 2')
    grad = gradient_2d_grid(x,y,qty,indexing=indexing)
    norm_grad = sm.norm(0,grad[0],grad[1])
    i,j = np.where(norm_grad<threshold)
    if len(i)==0 :
        raise ValueError('No potential saddle points has been found. Should increase the threshold or verify the indexing')
    else : 
        return i,j

def find_potential_saddle_points_with_hessian(i,j,x,y,qty,n=3, threshold=1e-1,rlim_sdl=8):
    i_sdls = []
    j_sdls = []
    qty_sdls = []
    for k in range(len(i)):
        matrice_nn = qty[i[k]-n:i[k]+n+1,j[k]-n:j[k]+n+1]
        if matrice_nn.shape!=(2*n+1,2*n+1):
            continue
        eig1 ,eig2 =  hessian_matrix_eigvals(hessian_matrix(matrice_nn, sigma=np.std(matrice_nn)*0.01, order='rc'))
        if (np.sum((eig1 *eig2)<0) >0.95*matrice_nn.size) & (sm.norm(0,x[i[k],j[k]],y[i[k],j[k]])<=rlim_sdl):
            i_sdls.append(i[k])
            j_sdls.append(j[k])
            qty_sdls.append(qty[i[k],j[k]])
    if len(qty_sdls)==0 :
        raise ValueError('No saddle point has been found. Try to increase the threshold or rlim_sdl, or try to decrease n.')
    else :
        return i_sdls, j_sdls, qty_sdls 

def find_potential_max_points_with_hessian(i,j,x,y,qty,n=3,rlim_max=8):
    i_sdls = []
    j_sdls = []
    qty_sdls = []
    for k in range(len(i)):
        matrice_nn = qty[i[k]-n:i[k]+n+1,j[k]-n:j[k]+n+1]
        if matrice_nn.shape!=(2*n+1,2*n+1):
            continue
        eig1 ,eig2 =  hessian_matrix_eigvals(hessian_matrix(matrice_nn, sigma=np.std(matrice_nn)*0.01, order='rc'))
        if (np.sum((eig2)<0) >0.95*matrice_nn.size) & (np.sum((eig1 )<0) >0.95*matrice_nn.size)  & (sm.norm(0,x[i[k],j[k]],y[i[k],j[k]])<=rlim_max):
            i_sdls.append(i[k])
            j_sdls.append(j[k])
            qty_sdls.append(qty[i[k],j[k]])
    if len(qty_sdls)==0 :
        raise ValueError('No maximum points has been found. Try to increase the threshold or rlim_max, or try to decrease n.')
    else :
        return i_sdls, j_sdls, qty_sdls     
    
def select_max_from_interest_points(i_sdls, j_sdls, qty_sdls):
    argmax_qty = np.argmax(qty_sdls)
    i_sdl = i_sdls[argmax_qty] 
    j_sdl = j_sdls[argmax_qty] 
    return i_sdl,j_sdl
    
def find_saddle_point_with_hessian(x,y,qty,n=2, threshold=1e-1,rlim_sdl=8,indexing='xy',verbose=False):
    i,j = find_potential_saddle_and_extremum_points(x,y,qty,n=n, threshold=threshold,indexing=indexing)
    i_sdls, j_sdls, qty_sdls = find_potential_saddle_points_with_hessian(i,j,x,y,qty,n=n, threshold=threshold, rlim_sdl=rlim_sdl)
    i_sdl, j_sdl = select_max_from_interest_points(i_sdls, j_sdls, qty_sdls)
    if verbose :
        print(len(qty_sdls),'Potential saddle points have been found')
        print('Chosen saddle point : (y,z) = (',  round(x[i_sdl,j_sdl],2),',',round(y[i_sdl,j_sdl],2),')' )
    return x[i_sdl,j_sdl],y[i_sdl,j_sdl]

def find_max_point_with_hessian(x,y,qty,n=2, threshold=1e-1,rlim_max=8,indexing='xy',verbose=False):
    i,j = find_potential_saddle_and_extremum_points(x,y,qty,n=n, threshold=threshold,indexing=indexing)
    i_sdls, j_sdls, qty_sdls = find_potential_max_points_with_hessian(i,j,x,y,qty,n=n,  rlim_max=rlim_max)
    i_sdl, j_sdl = select_max_from_interest_points(i_sdls, j_sdls, qty_sdls)
    if verbose :
        print(len(qty_sdls),'Potential saddle points have been found')
        print('Chosen saddle point : (y,z) = (',  round(x[i_sdl,j_sdl],2),',',round(y[i_sdl,j_sdl],2),')' )
    return x[i_sdl,j_sdl],y[i_sdl,j_sdl]

def make_hessian_e2_vector(x,y,qty):
    Hrr, Hrc, Hcc  = hessian_matrix(qty )
    mat = np.zeros((len(x), len(y), 2, 2))
    mat[:,:,0,0] = Hrr
    mat[:,:,1,0] = Hrc
    mat[:,:,0,1] = Hrc
    mat[:,:,1,1] = Hcc
    hess_val, hess_vec = np.linalg.eigh(mat)
    return hess_vec[:,:,:,1]
    
def make_linear_interpolator(x,y,qty):
    arr2d = reshape_to_2Darrays([x,y,qty])[0]
    return  LinearNDInterpolator(arr2d[:,:2],arr2d[:,-1])

def make_hessian_e2_interpolator(x,y,qty,indexing='xy'):
    if indexing == 'xy' :
        yaxis=0
        zaxis=1
    else : 
        yaxis=1
        zaxis=0
    e2 = make_hessian_e2_vector(x,y,qty)
    e2x = make_linear_interpolator(x,y,e2[:,:,yaxis])
    e2y = make_linear_interpolator(x,y,e2[:,:,zaxis])
    return e2x,e2y

def make_gradient_interpolator(x,y,qty,indexing='xy'):
    grad= gradient_2d_grid(x,y,qty,indexing=indexing)
    gx = make_linear_interpolator(x,y,grad[0])
    gy = make_linear_interpolator(x,y,grad[1])
    return gx,gy

def structure_tensor_eig_vec(qty,sigma=0.5,rho=1,indexing='xy'):
    if indexing == 'ij' :
        yaxis=0
        zaxis=1
    else : 
        yaxis=1
        zaxis=0

    S = structure_tensor_2d(qty, sigma, rho)
    st_val, st_vec = eig_special_2d(S)
    return st_vec[yaxis] ,st_vec[zaxis]

def make_structure_tensor_vec_interpolator(x,y,qty,sigma=0.5,rho=1,indexing='xy'):
    vec = structure_tensor_eig_vec(qty,sigma=sigma,rho=rho,indexing=indexing)
    st_x = make_linear_interpolator(x,y,vec[0])
    st_y = make_linear_interpolator(x,y,vec[1])
    return st_x,st_y



def outofbounds(t,pos,interxy,rlim): 
    
    if np.sqrt(pos[0]**2+pos[1]**2)>rlim:
        v=0
    else :
        v = 1       
    return v

outofbounds.terminal=True

     
def find_saddle_point(x,y,qty,n=3, threshold=1e-1,rlim_sdl=8,indexing='xy',verbose=True):
    x0,y0=np.nan,np.nan
    for t in np.arange(threshold/20,threshold,threshold/20):
        try:
            x0,y0=find_saddle_point_with_hessian(x,y,qty,n=n, threshold=t,rlim_sdl=rlim_sdl,indexing=indexing,verbose=False)
            break
        except:
            pass
    if np.isnan(x0):
        raise ValueError('No saddle point has been found. Try to increase the threshold or rlim_sdl, or try to decrease n.')
        
    if verbose==True:
        print('Saddle point : (y0,z0) = (',  round(x0,2),',',round(y0,2),')' )
    return x0,y0


def find_max_point(x,y,qty,n=3, threshold=1e-1,rlim_max=8,indexing='xy',verbose=True):
    x0,y0=np.nan,np.nan
    for t in np.arange(threshold/20,threshold,threshold/20):
        try:
            x0,y0=find_max_point_with_hessian(x,y,qty,n=n, threshold=t,rlim_max=rlim_max,indexing=indexing,verbose=False)
            break
        except:
            pass
    if np.isnan(x0):
        raise ValueError('No maximum point has been found. Try to increase the threshold or rlim_max, or try to decrease n.')
        
    if verbose==True:
        print('Maximum point : (y0,z0) = (',  round(x0,2),',',round(y0,2),')' )
    return x0,y0

def get_line_from_with_hess2(interhess2,
             x0 = 0,
             y0 = 0,
             t0 = 0,
             tfinal = 100,
             fac= 0.1,
             max_step=0.05,first_step=0.05,rlim=15,
             outofbounds = outofbounds):

    
    def vel(t,      # pseudo time
            pos,    # x and y positions
            interhess2,rlim): # eigenvector interpolators in x and y directions:   # some arbitrary magnification coef
        vv= [fac*(interhess2[0](pos[0], pos[1])),
             fac*(interhess2[1](pos[0], pos[1]))]

        return vv
    
 
    return solve_ivp(vel,[t0, tfinal],[x0, y0],
                    args=(interhess2,rlim ),
                    method="BDF", events=outofbounds,max_step=max_step,first_step=max_step).y


def outofbounds_with_grad(t,pos,interxy,rlim): 
    if np.sqrt(pos[0]**2+pos[1]**2)>rlim:
        v=0
    elif sm.norm(0,interxy[0](pos[0], pos[1]),interxy[1](pos[0], pos[1]))<1e-5:
        v=0
    else :
        v = 1       
    return v

outofbounds_with_grad.terminal=True

def get_line_with_gradient(intergrad,interhess2,
             x0 = 0,
             y0 = 0,
             t0 = 0,
             tfinal = 100,
             fac= 0.1,
             max_step=0.05,first_step=0.05,rlim=15,
             outofbounds = outofbounds_with_grad):

    
    def vel(t,      # pseudo time
            pos,    # x and y positions
            intergrad,rlim): # eigenvector interpolators in x and y directions:   # some arbitrary magnification coef
        vv= [(intergrad[0](pos[0], pos[1])),
             (intergrad[1](pos[0], pos[1]))]
        return vv
    
    dx,dy= interhess2[0]([x0,y0])[0],interhess2[1]([x0,y0])[0]
 
    return solve_ivp(vel,[t0, tfinal],[x0+(fac*dx), y0+(fac*dy)],
                    args=(intergrad,rlim ),
                    method="BDF", events=outofbounds_with_grad,max_step=max_step,first_step=max_step).y
def f_bmsh_pool(p0):
    b = interpBmsh.predict(p0)
    return b

def f_bmsp_pool(p0):
    b = interpBmsp.predict(p0)
    return b

def f_nmsh_pool(p0):
    b = interpNpmsh.predict(p0)
    return b

def f_nmsp_pool(p0):
    b = interpNpmsp.predict(p0)
    return b

def f_vmsh_pool(p0):
    b = interpVmsh.predict(p0)
    return b

def f_vmsp_pool(p0):
    b = interpVmsp.predict(p0)
    return b

def find_critic_point(x,y,qty,threshold=0.1,n=3,rlim_critic=8,indexing='xy',verbose=True):
    
    saddle_pt =False
    max_pt =False
    try :
        x0,y0= find_saddle_point(x, y, qty, n=n, threshold=threshold,rlim_sdl=rlim_critic,indexing=indexing,verbose=False)
        saddle_pt =True
    except:
        
        try :
            x0,y0= find_max_point(x, y, qty, n=n, threshold=threshold,rlim_max=rlim_critic,indexing=indexing,verbose=False)
            max_pt =True
        except:
            raise ValueError('no critic point was found')
    if verbose :
        if saddle_pt:
            print('Saddle point : (y0,z0) = (',  round(x0,2),',',round(y0,2),')')
        if max_pt:
            print('Maximun point : (y0,z0) = (',  round(x0,2),',',round(y0,2),')') 
                  
            
    return x0,y0,saddle_pt,max_pt

def make_half_line(intergrad,interhess2,x0,y0,fac_e2,rlim=15,twrd_yext=False):   #follow_null_grad_twrd_ext=False,follow_decreasing_grad=False      
    part1 = get_line_with_gradient(intergrad,interhess2,x0=x0,y0=y0,fac=fac_e2,rlim=rlim)
    r0= sm.norm(0,part1[0][-1],part1[1][-1])
    if r0<rlim:
        new_x0,new_y0 = part1[0][-1],part1[1][-1]
        dx,dy = part1[0][-3]-part1[0][-1],part1[1][-3]-part1[1][-1]
        hx,hy = interhess2[0](new_x0,new_y0),interhess2[1](new_x0,new_y0)
        
        #rh= sm.norm(0,part1[0][-1]+hx,part1[1][-1]+hy)
        if twrd_yext:
            if (np.sign(hx)!=np.sign(new_x0)):
                part2= get_line_from_with_hess2(interhess2,x0=new_x0,y0=new_y0,fac=-1,rlim=rlim)
            elif (np.sign(hx)==np.sign(new_x0)): 
                part2= get_line_from_with_hess2(interhess2,x0=new_x0,y0=new_y0,fac=1,rlim=rlim)
            else:
                part2 = [np.nan,np.nan]
        #print(((np.sign(dx)==np.sign(hx)) ,  (np.sign(dy)==np.sign(hy))) , (r0>rh), ((np.sign(new_x0)==np.sign(hx)) ,  (np.sign(new_y0)==np.sign(hy)))  )
        else:
            if (np.sign(dx)==np.sign(hx)) &  (np.sign(dy)==np.sign(hy))  :
                part2= get_line_from_with_hess2(interhess2,x0=new_x0,y0=new_y0,fac=-1,rlim=rlim)
            elif (np.sign(dx)!=np.sign(hx)) &  (np.sign(dy)!=np.sign(hy)):
                part2= get_line_from_with_hess2(interhess2,x0=new_x0,y0=new_y0,fac=1,rlim=rlim)
            else :
                part2 = [[np.nan],[np.nan]]
        if np.min(part2[0])<np.min(part1[0]):
            half_line = np.concatenate([part2[0][::-1],part1[0][::-1]]),np.concatenate([part2[1][::-1],part1[1][::-1]])
        else: 
            half_line = np.concatenate([part1[0],part2[0]]),np.concatenate([part1[1],part2[1]])
    else: 
        half_line = part1
    return half_line
    

def find_max_line(x,y,qty0,n=2, threshold=1e-1,rlim_critic=8, rlim=15, fac_e2=0.1,indexing='xy',twrd_yext=False,norm_qty=True,verbose=True):
    if norm_qty:
        qty = qty0*10/np.median(qty0)
    else:
        qty=qty0
    x0,y0,saddle_pt,max_pt = find_critic_point(x, y, qty, n=n, threshold=threshold,rlim_critic=rlim_critic,indexing=indexing,verbose=verbose)
    interhess2 = make_hessian_e2_interpolator(x,y,qty)
    intergrad = make_gradient_interpolator(x,y,qty,indexing=indexing)

    part1 = make_half_line(intergrad,interhess2,x0,y0,-fac_e2,rlim=rlim,twrd_yext=twrd_yext)
    part2 = make_half_line(intergrad,interhess2,x0,y0,fac_e2,rlim=rlim,twrd_yext=twrd_yext)
    rend_part1 = sm.norm(0,part1[0][-1],part1[1][-1])
    rbegin_part1 = sm.norm(0,part1[0][0],part1[1][0])
    rend_part2 = sm.norm(0,part2[0][-1],part2[1][-1])
    rbegin_part2 = sm.norm(0,part2[0][0],part2[1][0])
    if (rend_part1>rbegin_part1) & (rend_part2>rbegin_part2) :
        part2 =np.asanyarray(part2)
        part2[0] =part2[0][::-1]
        part2[1] =part2[1][::-1]
    elif (rend_part1<rbegin_part1) & (rend_part2<rbegin_part2) :
        part2 =np.asanyarray(part2)
        part2[0] =part2[0][::-1]
        part2[1] =part2[1][::-1]
        
    
    #rend_part1 = sm.norm(0,part1[0][-1],part1[1][-1])
    rbegin_part1 = sm.norm(0,part1[0][0],part1[1][0])
    #rend_part2 = sm.norm(0,part2[0][-1],part2[1][-1])
    rbegin_part2 = sm.norm(0,part2[0][0],part2[1][0])
    if rbegin_part2>rbegin_part1:
        max_line = np.concatenate([part2[0],part1[0]]),np.concatenate([part2[1],part1[1]])
    #elif (np.sign(part2[0][1]-part2[0][0])<0) &  sm.norm(0,part2[0][-1],part2[1][-1])>rbegin_part1 :
    #    max_line = np.concatenate([part2[0],part1[0]]),np.concatenate([part2[1],part1[1]])
    else: 
        max_line = np.concatenate([part1[0],part2[0]]),np.concatenate([part1[1],part2[1]])
    
    #max_line = np.concatenate([part1[0],part2[0]]),np.concatenate([part1[1],part2[1]])
    return max_line


def make_vectorial_qty_msp(knn_function,coord,regular_coord=None,sigma=20,rm_normal=True):
    qtyx,qtyy,qtyz = smap.compute_knn(knn_function,np.array(coord).T).T
    if regular_coord is not None:
        qtyx,qtyy,qtyz = smap.interpolate_on_regular_grid(coord[1],coord[2],[qtyx,qtyy,qtyz],regular_coord[1],regular_coord[2])
    else:
        regular_coord[0],regular_coord[1],regular_coord[2] = coord[0],coord[1],coord[2]
    if sigma !=0:
        qtyx,qtyy,qtyz = smap.make_gaussian_filter([qtyx,qtyy,qtyz],(sigma,sigma))
    if rm_normal :
        qtyx,qtyy,qtyz = smap.remove_normal_to_shue98(regular_coord[0],regular_coord[1],regular_coord[2],qtyx,qtyy,qtyz)
    return qtyx,qtyy,qtyz

def make_scalar_qty_msp(knn_function,coord,regular_coord=None,sigma=20):
    qty = smap.compute_knn(knn_function,np.array(coord).T).T
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
    R= (0.1*(b2*b1)**(3/2))/(np.sqrt(cst.mu_0)*np.sqrt(b2*np1+b1*np2)*np.sqrt(b1+b2)) *1e3
    return R

def make_RR_with_shear_flow(npmsp,npmsh,bmsp,bmsh,vmsp,vmsh,alpha,bimf_norm=5,vsw_norm=400):   
    b1 = sm.norm(bmsp[0],bmsp[1],bmsp[2])*1e-9*np.sin(np.radians(alpha)/2)
    b2 = sm.norm(bmsh[0],bmsh[1],bmsh[2])*1e-9*bimf_norm*np.sin(np.radians(alpha)/2)
    v1 = sm.norm(vmsp[0],vmsp[1],vmsp[2])*1e3*np.sin(np.radians(alpha)/2)
    v2 = sm.norm(vmsh[0],vmsh[1],vmsh[2])*1e3*vsw_norm*np.sin(np.radians(alpha)/2)
    np1 = npmsp
    np2 = npmsh
    R= (0.1*(b2*b1)**(3/2))/(np.sqrt(cst.mu_0)*np.sqrt(b2*np1+b1*np2)*np.sqrt(b1+b2)) *1e3
    ca= np.sqrt((b1*b2*(b1+b2))/(cst.mu_0*(b2*np1+b1*np2)))
    vshear = (v2-v1)/2
    A = (vshear/ca)**2
    B = 4*np1*b2*np2*b1
    C = (b2*np1+b1*np2)**2
    Rv= R*(1-A*B/C)
    return Rv

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
