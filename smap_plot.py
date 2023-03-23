import sys
import pandas as pd
import numpy as np

sys.path.append('./space')
sys.path.append('Python_functions')
sys.path.append('./')

from scipy import constants as cst
from skimage.feature import peak_local_max
from space.models import planetary as smp
from space import smath as sm
import matplotlib.pyplot as plt
from space.coordinates import coordinates as scc
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
msh = smp.Magnetosheath(magnetopause='mp_shue1998', bow_shock ='bs_jelinek2012')

import shear_maps as smap



########################################Fonction de plot##########################################

def plot_the_map(jj,yy,zz,clock,computer_name):
    new_cla=clock
    fig,axs=fig_nxm_plots(1,1)

    if computer_name=='JComputer':
        norm=80
        title='Current Density'
    elif computer_name=='RComputer':
        norm=0.35
        title='Reconnection Rate'
    elif computer_name=='SAComputer':
        norm=180
        title='Shear Angle'
        
    im = axs.pcolormesh(yy,zz, jj,  norm=Normalize(0,norm), cmap='jet',shading='auto')
    fig.colorbar(im,ax=axs)    
    axs.set_title(title)
    axs.arrow(0,0,10*np.sin(np.radians(new_cla)),10*np.cos(np.radians(new_cla)),color='dimgrey',head_width=0.5,zorder=10)
    axs.arrow(0,0,-10*np.sin(np.radians(new_cla)),-10*np.cos(np.radians(new_cla)),color='dimgrey',head_width=None,zorder=10)
    fig.tight_layout()

    
def fig_nxm_plots(n=2,m=2,norm=Normalize(-60,60),cmap='seismic',msh=msh):
    
    mp,bs = msh.boundaries(np.pi/2,np.linspace(0,2*np.pi,50))

    fig,axs = plt.subplots(m,n,figsize=(n*7.,m*6+0.25))
    #for ax in axs.ravel():
    axs.plot(mp[1],mp[2],ls='-.',color='k')
    axs.plot(bs[1],bs[2],ls='--',color='k')
    axs.set_aspect('equal')
 
    axs.axhline(0,ls='--',c='k',alpha=0.75)
    axs.axvline(0,ls='--',c='k',alpha=0.75)
    axs.set_xlim(-16,16)
    axs.set_ylim(-16,16)
    
    alphabet = list('abcdefghijklmnopqr')
    apair= alphabet[::2] 
    aimpair= alphabet[1::2] 
    fig.suptitle('   ')

    fig.tight_layout()

    return fig,axs 



def find_max_point(x,y,qty,n=3,rlim_max=9,min_distance=5,indexing='xy',verbose=True):
    
    coord = peak_local_max(qty, min_distance=min_distance)
    xm = np.asarray([x[c[0],c[1]] for c in coord])
    ym = np.asarray([y[c[0],c[1]] for c in coord])
    qm = np.asarray([qty[c[0],c[1]] for c in coord])

    r = sm.norm(xm,ym,0)
    xm ,ym ,qm = xm[r<=rlim_max], ym[r<=rlim_max],qm[r<=rlim_max]
    
    return xm[qm.argmax()],ym[qm.argmax()]


def make_reg_grid():
    
    N=401
    th = np.linspace(0,0.95*np.pi,N)#
    ph =np.linspace(0,2*np.pi,2*N)
    
    theta,phi = np.meshgrid(th,ph)
    Xmp,Ymp,Zmp = msh.magnetopause(theta,phi)
    yy,zz=smap.make_regular_grid(xlim=(-22,22),ylim=(-22,22),nb_pts=N,indexing="xy")
    xx = smap.interpolate_on_regular_grid(Ymp,Zmp,[Xmp],yy,zz)[0]
    
    theta,phi = scc.cartesian_to_spherical(xx,yy,zz)[1:]
    mp,bs = msh.boundaries(np.pi/2,np.linspace(0,2*np.pi,50))
    
    return ([xx, yy, zz],[Xmp, Ymp, Zmp])

###################################Make_values(à décaler dans le computer)###############################################

def making_values(computer, grids, angles, regular_coord = None):
    Bmsh, bmspgrids = grids[0], grids[1]
    cone, clock, tilt = angles[0], angles[1], angles[2]
    xx, yy, zz = regular_coord
    if type(computer).__name__=='RComputer':
            npmspgrids=computer.npmspgrids(cone, tilt)
            npmshgrids=computer.npmshgrids(cone, tilt)
            
            Npmsh = transform_scalar_qty_msh_swi(npmshgrids, regular_coord=[xx,yy,zz], new_clock=np.radians(clock))
            qty=computer.make_values(Bmsh,Npmsh,[bmspgrids,npmspgrids], cone, clock, regular_coord=[xx,yy,zz])
    
    elif type(computer).__name__=='JComputer': 
        qty=computer.make_values(Bmsh,bmspgrids, cone, clock, regular_coord=[xx,yy,zz])
        
    elif type(computer).__name__=='SAComputer':
        qty=computer.make_values(Bmsh,bmspgrids, cone, clock, regular_coord=[xx,yy,zz])
    
    else:
        print('problème dans comparaison classes')
    return(qty)

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
        

def traitement(yy,zz,jj):
    x0,y0 = find_max_point(yy,zz,jj)
    mp,bs = msh.boundaries(np.pi/2,np.linspace(0,2*np.pi,50))
    interhess2 = smap.make_hessian_e2_interpolator(yy,zz,jj)
    part1 =smap.get_line_from_with_hess2(interhess2,x0=x0,y0=y0,fac=-1,rlim=sm.norm(mp[0],mp[1][0],mp[2][0]))
    part2 =smap.get_line_from_with_hess2(interhess2,x0=x0,y0=y0,fac=1,rlim=sm.norm(mp[0],mp[1][0],mp[2][0]))
    max_current_line = np.concatenate([part1[0][::-1],part2[0]]),np.concatenate([part1[1][::-1],part2[1]])
    return max_current_line     
    
    

################################################Interpolation############################################################
                #vec, coord
def swi_to_pgsm(   grids, cone=None, new_clock=None, regular_coord=None):   #interpolation for msh
    
    Qty_list=[]   
    sigma=20
    old_clock = np.radians(90)
    coord1, coord2, coord3, qtyx, qtyy, qtyz=grids.values()
    coord=[coord1, coord2, coord3]
    qtyx,qtyy,qtyz=qtyx,qtyy,qtyz
    rm_normal=True
    
    if np.sign(cone)==-1:
        qtyx,qtyy,qtyz  = smap.swi_to_negative_bximf(coord[1],coord[2],qtyx,qtyy,qtyz)

    if new_clock is not None:
        xr,yr,zr,qtyx,qtyy,qtyz = smap.rotates_clock_angle(coord[0],coord[1],coord[2],qtyx,qtyy,qtyz,new_clock,old_clock)
 
    else :
        xr,yr,zr = coord[0],coord[1],coord[2]

    if regular_coord is not None:
        qtyx,qtyy,qtyz = smap.interpolate_on_regular_grid(yr,zr,[qtyx,qtyy,qtyz],regular_coord[1],regular_coord[2])
 
    else:
        regular_coord[0],regular_coord[1],regular_coord[2] = xr,yr,zr

    if sigma !=0:
        qtyx,qtyy,qtyz = smap.make_gaussian_filter([qtyx,qtyy,qtyz],(sigma,sigma))
    
    if rm_normal :
        qtyx,qtyy,qtyz = smap.remove_normal_to_shue98(regular_coord[0],regular_coord[1],regular_coord[2],qtyx,qtyy,qtyz)
 
      
    Qty_list.append(qtyx) #On obtient à la fin une liste des quantités: Bx,By,Bz,Vx,Vy,Vz . Il y a normalement toujours les clés à la fin
    Qty_list.append(qtyy)
    Qty_list.append(qtyz)
    return Qty_list

def transform_scalar_qty_msh_swi(grids,regular_coord=None, new_clock=None):
    
    Qty_list=[]   
    sigma=20
    old_clock = np.radians(90)
    coord1, coord2, coord3, qty=grids.values()
    coord=[coord1, coord2, coord3]
    
    if new_clock is not None:
        xr,yr,zr = smap.rotates_phi_angle(coord[0],coord[1],coord[2],new_clock - old_clock)
        #regu('rotates but for scalar qty'+datetime.datetime.now())
    else:
        xr,yr,zr= coord
        #print('rotates but for scalar qty'+datetime.datetime.now())

    if regular_coord is not None:
        qty = smap.interpolate_on_regular_grid(yr,zr,[qty],regular_coord[1],regular_coord[2])[0]
        #print('checked coordonates'+datetime.datetime.now())
    else :
        qty = smap.interpolate_on_regular_grid(yr,zr,[qty],coord[1],coord[2])[0]
        #print('checked coordonates'+datetime.datetime.now())
    if sigma!=0:
        qty = smap.make_gaussian_filter([qty],(sigma,sigma))[0]
        #print('Gaussian filter for sigma'+datetime.datetime.now())
    return qty

def qty(computer):
        if type(computer).__name__=='RComputer':
            return ('R')
        if type(computer).__name__=='JComputer':
            return('J')
        if type(computer).__name__=='SAComputer':
            return('SA')
##################################Classes de computer##################################################

class JComputer:
    def __init__(self):
        print("Init J")

    def bmshgrids(self, cone, tilt):
        return pd.read_pickle(f'grid_b_msh_{cone}.pkl')
        
    def bmspgrids(self, cone, tilt):
        return pd.read_pickle(f'grid_b_msp_{cone}_{tilt}.pkl')
    
    def make_values(self, Bmsh, msp_grids, cone, clock, regular_coord=None):
    
        bxmsh,bymsh,bzmsh=Bmsh[0],Bmsh[1],Bmsh[2]
        x,y,z,bxmsp,bymsp,bzmsp=msp_grids.values()
        [xx,yy,zz]=regular_coord
        
        jj = make_current_density(xx,yy,zz,bxmsp,bymsp,bzmsp,bxmsh,bymsh,bzmsh,bimf_norm=5.0, dmp = 800/6400)[-1]
        
        return jj
    


class RComputer:
    def __init__(self):
        print("Init R")
        
    def bmshgrids(self, cone, tilt):
        return pd.read_pickle(f'grid_b_msh_{cone}.pkl')
        
    def bmspgrids(self, cone, tilt):
        return pd.read_pickle(f'grid_b_msp_{cone}_{tilt}.pkl')
    
    def npmshgrids(self, cone, tilt):
        return pd.read_pickle(f'grid_np_msh_{cone}.pkl')
        
    def npmspgrids(self, cone, tilt):
        return pd.read_pickle(f'grid_np_msp_{cone}_{tilt}.pkl')
    
    def make_values(self, bmsh, npmsh, msp_grids, cone, clock, regular_coord=None):
        bxmsh,bymsh,bzmsh = bmsh[0],bmsh[1],bmsh[2]
        x,y,z,bxmsp,bymsp,bzmsp = msp_grids[0].values()
        x1,y1,z1,npmsp = msp_grids[1].values()

        bmsp=[bxmsp,bymsp,bzmsp]
        [xx,yy,zz]=regular_coord
    
        alpha = smap.shear_angle(bxmsp,bymsp,bzmsp,bxmsh,bymsh,bzmsh)
        R = make_RR(npmsp,npmsh,bmsp,bmsh,alpha,bimf_norm=5.0)
        return R 
        

    
class SAComputer:
    def __init__(self):
        print("Init SA")
        
    def bmshgrids(self, cone, tilt):
        return pd.read_pickle(f'grid_b_msh_{cone}.pkl')
        
    def bmspgrids(self, cone, tilt):
        return pd.read_pickle(f'grid_b_msp_{cone}_{tilt}.pkl')
    
    def make_values(self, Bmsh, msp_grids, cone, clock, regular_coord=None):
    
        bxmsh,bymsh,bzmsh=Bmsh[0],Bmsh[1],Bmsh[2]
        x,y,z,bxmsp,bymsp,bzmsp=msp_grids.values()
        [xx,yy,zz]=regular_coord
        
        alpha = smap.shear_angle(bxmsp,bymsp,bzmsp,bxmsh,bymsh,bzmsh)        
        return alpha

###########################################Fonction globale de MakeMap###################################################
def make_map(clock, cone, tilt, computer):#,path)
    
    qty_name=qty(computer)
    [xx,yy,zz]=make_reg_grid()[0]
    print('I made a regular grid')
    
    bmspgrids=computer.bmspgrids(cone, tilt)
    bmshgrids=computer.bmshgrids(cone, tilt)
    print('I imported grids')
    
    Bmsh = swi_to_pgsm(bmshgrids, cone, clock, regular_coord=[xx, yy, zz])  #Voir si on se débarrasse des grilles régulières
    print('I rotated and interpolated')
    
    qty=making_values(computer, [Bmsh,bmspgrids], [cone, clock, tilt], regular_coord=[xx,yy,zz])
    print('Jai créé les valeurs de densité')
    #Valeurs obtenues à enregistrer qqpart: pd.to_pickle({'y':yy,'z':zz,'qty':qty},f'{path}/map_{qty_name}_{clock}_{cone}_{tilt}.pkl')
    
    return plot_the_map(qty,yy,zz,clock,type(computer).__name__)

















