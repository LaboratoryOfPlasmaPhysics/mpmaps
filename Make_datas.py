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
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from datetime import timedelta, datetime
from glob import glob
import os

import sys
sys.path.append('./space/space')

from coordinates import coordinates as scc

sys.path.append('Notebooks/space')
sys.path.append('.')

#########################THEMIS##########################################################################################
def make_dataset_themis(start,stop):
    satellites = ["A","B","C","D","E"]
    
    for sat in satellites:
        print(sat)
        make_data_themis(sat,start,stop)
        make_pos_themis(sat,start,stop)

def make_pos_themis(sat):
    pos = pd.read_pickle(f'/DATA/michotte/Datasets/THEMIS/raw/TH{sat}_OR_SSC_XYZ_GSM_{start}_{stop}.pkl')
    pos.columns = ['X','Y','Z']
    pos = pos[~pos.index.duplicated(keep='first')]
    pos = pos.sort_index()
    pos = pos.resample('5S').asfreq()
    pos = pos.interpolate(limit=int(12*5),limit_direction='both',center=True)
    pos['R'],pos['theta'],pos['phi'] = scc.cartesian_to_spherical(pos.X,pos.Y,pos.Z)
    pos = pos['01-09-2007':]
    pos.to_pickle(f'/DATA/michotte/Datasets/THEMIS/Datasets_THEMIS/TH{sat}_pos_GSM_5S_{pos.index[0].year}_{pos.index[-1].year}.pkl')
    del pos
    
def make_data_themis(sat,start,stop):
    produits =["density",'velocity',"t3"]
    names = [["Np"],["Vx","Vy","Vz"],["Tperp1","Tperp2","Tpara"]]
    columns_order = ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']
    data = pd.DataFrame()
    for p,n in zip(produits,names):

        mome = pd.read_pickle(glob(f'/DATA/michotte/Datasets/THEMIS/raw/TH{sat}_L2_MOM_th{sat.lower()}_peim*{p}*{start}_{stop}.pkl')[0]).resample('5S').mean()
        #mome2 = pd.read_pickle(glob(f'/DATA/michotte/Datasets/THEMIS/raw/TH{sat}_L2_MOM_th{sat.lower()}_peim*{p}*2021_2022.pkl')[0]).resample('5S').mean()
        mome.columns = n

        df = mome.copy()

        data = pd.concat([data,df],axis=1)
        if data.index.is_monotonic==False:
                raise ValueError("index redondant")
        del mome, df
    df = pd.read_pickle(glob(f'/DATA/michotte/Datasets/THEMIS/raw/TH{sat}_L2_FGM_th{sat.lower()}*{start}_{stop}.pkl')[0]).resample('5S').mean()

    df.columns = ['Bx', 'By', 'Bz']

    data = pd.concat([data,df],axis=1)
    del df
    data['Tp'] = (data['Tpara'] +  data['Tperp1'] + data['Tperp2'] )/3 * 1.160451812e4 
    data = data[columns_order]

    data[(data.Np)<=0]=np.nan
    data[(data.Tp)<=0]=np.nan
    #data[(data.Tp)>=1e10]=np.nan
    data[abs(data.Vx)>=2000]=np.nan
    data[abs(data.Vy)>=2000]=np.nan
    data[abs(data.Vz)>=2000]=np.nan
    for c in columns_order :
        data[c] = medfilt(data[c],3)
    data = data['01-09-2007':]
    data.to_pickle(f'/DATA/michotte/Datasets/THEMIS/Datasets_THEMIS/TH{sat}_data_GSM_5S_{data.index[0].year}_{data.index[-1].year}.pkl')
    del data

######################## Cluster ##########################################################################################
def make_dataset_cluster(start,stop):
    
    #Creation of data:
    dataC1=data_creation_C1(start,stop)
    dataC3=data_creation_C3(start,stop)
    
    #temperature
    dataC1['Tp'] = (dataC1['Tpara'] + 2 * dataC1['Tperp'])/3 * 1.160451812e4
    dataC1=conditions_on_temperature(dataC1) #Commun à d'autres sat
    
    dataC3['Tp'] = (dataC3['Tpara'] + 2 * dataC3['Tperp'])/3 * 1.160451812e4 
    dataC3=conditions_on_temperature(dataC3)
        
    #Enregistrement
    save_data_cluster(dataC1,dataC3)
    del dataC1
    del dataC3
    
    #Position
    make_pos_cluster(start,stop)
    
    #Old datasets
    status_C1()
    status_C3()

def data_creation_C1(start,stop):
    products  =  ["c1_b_gsm", "c1_hia_dens", "c1_hia_v_gsm", "c1_hia_tpar" ,"c1_hia_tperp" ]
    names = [['Bx','By','Bz'],  ['Np'] , ['Vx','Vy','Vz'], ['Tpara'], ['Tperp'] ]
    
    data = pd.DataFrame()
    for product,name in zip(products,names):
        df = pd.read_pickle(f'/DATA/michotte/Datasets/Cluster/raw/{product}_{start}_{stop}.pkl').resample('5S').mean()
        df.columns = name
        data = pd.concat([data,df],axis=1)
        del df
    return data

def data_creation_C3(start,stop):
    products  =  ["c3_b_gsm", "c3_hia_dens", "c3_hia_v_gsm", "c3_hia_tpar" ,"c3_hia_tperp" ]
    names = [['Bx','By','Bz'],  ['Np'] , ['Vx','Vy','Vz'], ['Tpara'], ['Tperp'] ]

    data = pd.DataFrame()
    for product,name in zip(products,names):
        df = pd.read_pickle(f'/DATA/michotte/Datasets/Cluster/raw/{product}_{start}_{stop}.pkl').resample('5S').mean()
        df.columns = name
        data = pd.concat([data,df],axis=1)
        del df
    return data

def save_data_cluster(dataC1,dataC3):
    columns_order = ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']
    
    for c in columns_order :
        dataC1[c] = medfilt(dataC1[c],3)
        dataC3[c] = medfilt(dataC3[c],3)
    dataC1.to_pickle(f'/DATA/michotte/Datasets/Cluster/Datasets_Cluster/C1_data_GSM_5S_{data.index[0].year}_{data.index[-1].year}.pkl')
    dataC3.to_pickle(f'/DATA/michotte/Datasets/Cluster/Datasets_Cluster/C3_data_GSM_5S_{data.index[0].year}_{data.index[-1].year}.pkl')
    


def make_pos_cluster(start,stop):

    C1 = pd.read_pickle(f'/DATA/michotte/Datasets/Cluster/raw/c1_xyz_gsm_{start}_{stop}.pkl')
    C3=pd.read_pickle(f'/DATA/michotte/Datasets/Cluster/raw/c3_xyz_gsm_{start}_{stop}.pkl')
    final_pos=[]
    
    for pos in [C1,C3]:
        pos = pos[~pos.index.duplicated(keep='first')]
        pos.columns = ["X","Y","Z"]
        pos = pos.resample('5S').asfreq()
        pos = pos.interpolate(limit=int(12*5),limit_direction='both',center=True)
        pos['R'],pos['theta'],pos['phi'] = scc.cartesian_to_spherical(pos.X,pos.Y,pos.Z)
        final_pos.append(pos)
        
        final_pos[0].to_pickle(f'/DATA/michotte/Datasets/Cluster/Datasets_Cluster/C1_pos_GSM_5S_{pos.index[0].year}_{pos.index[-1].year}.pkl')
        final_pos[1].to_pickle(f'/DATA/michotte/Datasets/Cluster/Datasets_Cluster/C3_pos_GSM_5S_{pos.index[0].year}_{pos.index[-1].year}.pkl')

def status_C1():
    status = pycdf.CDF('/DATA/michotte/Datasets/Cluster/raw/C1_CP_CIS_MODES__20010201_000000_20200101_000000_V210310.cdf')
    modes = status['cis_mode__C1_CP_CIS_MODES'][:]
    epoch  = status['time_tags__C1_CP_CIS_MODES'][:]
    status =pd.DataFrame({'modes' : modes},index=epoch)
    status = status.resample('5S').mean()
    status = status.interpolate(method='nearest',limit=12,center=True)
    status.to_pickle(f'/DATA/michotte/Datasets/Cluster/Datasets_Cluster/C1_modes_CIS_{status.index[0].year}_{status.index[-1].year}.pkl')
    

def status_C3():
    status = pycdf.CDF('/DATA/michotte/Datasets/Cluster/raw/C3_CP_CIS_MODES__20010201_000000_20100101_000000_V170421.cdf')
    modes = status['cis_mode__C3_CP_CIS_MODES'][:]
    epoch  = status['time_tags__C3_CP_CIS_MODES'][:]
    status =pd.DataFrame({'modes' : modes},index=epoch)
    status = status.resample('5S').mean()
    status = status.interpolate(method='nearest',limit=12,center=True)
    status.to_pickle(f'/DATA/michotte/Datasets/Cluster/Datasets_Cluster/C3_modes_CIS_{status.index[0].year}_{status.index[-1].year}.pkl')

####################### MMS ###############################################################################################
def make_dataset_MMS(start,stop):
    #Creation of data:
    data=data_creation(start,stop)
  
    ####################################################################################
    #V_GSE becomes V_GSM    
    v = data[['Vx','Vy','Vz']].copy()
    start = data.index[0]
    stop = data.index[-1]

    dt=timedelta(days=1)
    N= int(np.ceil((stop-start)/dt))

    intervals = [v[start+i*dt : start+(i+1)*dt ] for i in range(N) if len(v[start+i*dt : start+(i+1)*dt ])>1]

    print(N,str(start.year),start + N*dt)
    
    with Pool(30) as p:
        v_gsm = pd.concat( p.map(V_gse_to_gsm,intervals) )
        
    v_gsm = v_gsm[~v_gsm.index.duplicated(keep='last')].sort_index()     
    
    data['Vx'] = v_gsm['Vx']
    data['Vy'] = v_gsm['Vy']
    data['Vz'] = v_gsm['Vz']
    
    del v_gsm,v
    ###############################################################################################
    
    
    #temperature
    data['Tp'] = (data['Tpara'] + 2 * data['Tperp'])/3 * 1.160451812e4 
    data=conditions_on_temperature(data)
    
    #Enregistrement
    save_data_MMS(data)
    del data
    
    #Position
    make_pos_MMS(start,stop)
    
    #Old datasets
    old_datasets_MMS()

def read_data(start,stop):

    df = pd.read_pickle(f'/DATA/michotte/Datasets/MMS/raw/mms1_b_gsm_{start}_{stop}.pkl').resample('5S').mean()
    df.columns = ['Bx','By','Bz']
    data=pd.DataFrame()
    #dfb = pd.read_pickle(f'/DATA/michotte/Datasets/MMS/raw/mms1_b_gsm_2018_2021.pkl').resample('5S').mean()
    #dfb.columns = ['Bx','By','Bz']
    #data = pd.concat([df,dfb],axis = 0)
    data=pd.concat([df,dfb],axis = 1)
    return data

def data_creation(start,stop):
    
    data=read_data(start,stop)
    products  =  [ "mms1_dis_ni", "mms1_dis_vgse", "mms1_dis_tpara" ,"mms1_dis_tperp" ]
    names = [  ['Np'] , ['Vx','Vy','Vz'], ['Tpara'], ['Tperp'] ]
    
    for product,name in zip(products,names):
        df = pd.read_pickle(f'/DATA/michotte/Datasets/MMS/raw/{product}_{start}_{stop}.pkl').resample('5S').mean()
        df.columns = name
        data = pd.concat([data,df],axis=1)
        del df
    return data

def make_pos_MMS(start,stop):
    
    pos = pd.read_pickle(f'/DATA/michotte/Datasets/MMS/raw/mms1_orbit_gsm_{start}_{stop}.pkl')
    pos = pos*1e3/acst.R_earth
    pos = pos.resample('5S').asfreq()
    pos = pos.interpolate(limit=int(12*5),limit_direction='both',center=True)
    pos['R'],pos['theta'],pos['phi'] = scc.cartesian_to_spherical(pos.X,pos.Y,pos.Z)
    pos.to_pickle(f'/DATA/michotte/Datasets/MMS/datasets_MMS/MMS1_pos_GSM_5S_{pos.index[0].year}_{pos.index[-1].year}.pkl')

def V_gse_to_gsm(v):
    cvals = coord.Coords(v.values, 'GSE', 'car')
    cvals.ticks = Ticktock(v.index.values.astype(str), 'ISO')
    newcoord = cvals.convert('GSM', 'car').data
    return pd.DataFrame({'Vx' : newcoord[:,0],'Vy' : newcoord[:,1],'Vz' : newcoord[:,2] },index=v.index)

def conditions_on_temperature(data): 
    columns_order = ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']
    data = data[columns_order]
    data[abs(data.Bx)>2000]=np.nan
    data[abs(data.By)>2000]=np.nan
    data[abs(data.Bz)>2000]=np.nan
    data[(data.Np)>=500]=np.nan
    data[(data.Np)<=0]=np.nan
    data[(data.Tp)>=1e10]=np.nan
    data[abs(data.Vx)>1500]=np.nan
    data[abs(data.Vy)>1500]=np.nan
    data[abs(data.Vz)>1500]=np.nan
    return(data)

def save_data_MMS(data):
    for c in columns_order :
            data[c] = medfilt(data[c],3)
    data.to_pickle(f'/DATA/michotte/Datasets/MMS/datasets_MMS/MMS1_data_GSM_5S_{data.index[0].year}_{data.index[-1].year}.pkl')

def old_datasets_MMS():
    pos = pd.read_pickle(f'/DATA/nguyen/MMS/MMS1_pos.pkl')
    pos = pos.loc[pos.index.duplicated(keep='first')==False]
    pos = pos.resample('5S').mean()
    pos = pos.interpolate(limit=36,limit_area='inside',limit_direction='both',method='polynomial', order=2)

    pos = su.add_columns_to_df(pos,scc.cartesian_to_spherical(pos.X,pos.Y,pos.Z),['R','theta','phi'])
    pos.to_pickle(f'../Datasets/MMS/position/5S_V1/position_MMS1_2015_2019.pkl')
    del pos
    
    data = pd.read_pickle(f'/DATA/nguyen/MMS/MMS1.pkl')
    data['Tp'] = (data['Tpara']+2*data['Tperp'])/3 * 1.160451812e4
    data = data[['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']]
    data = data.resample('5S').mean()#.interpolate(limit=12,limit_area='inside',limit_direction='both')
    for x in data.columns:
        data[x] = medfilt(data[x], 3)
    data.to_pickle(f'data/5S_V1/data_MMS1_2015_2019.pkl')
    
    data = pd.read_pickle(f'data/5S_V1/data_MMS1_2015_2019.pkl') 
    for x in data.columns:
        data[x] = medfilt(data[x], 3)
    data.to_pickle(f'data/5S_V1/data_MMS1_2015_2019.pkl')
    
####################### DS ################################################################################################
def make_dataset_doublestar(start,stop):
    #Creation of data:
    data=data_creation_doublestar(start,stop)
    #temperature
    data['Tp'] = (data['Tpara'] + 2 * data['Tperp'])/3 * 1.160451812e4
    data=conditions_on_temperature(data) #Commun à d'autres sat
    #Enregistrement
    save_data_doublestar(data)
    del data
    #Position
    make_pos_doublestar(start,stop)
    #Old datasets
    old_datasets_doublestar()
    
def data_creation_doublestar(start,stop):
    products  =  ["ds1_b_gsm", "ds1_n", "ds1_v_gsm", "ds1_t" ]
    names = [['Bx','By','Bz'],  ['Np'] , ['Vx','Vy','Vz'], ['Tpara','Tperp'] ]
    
    data = pd.DataFrame()
    for product,name in zip(products,names):
        df = pd.read_pickle(f'/DATA/michotte/Datasets/DoubleStar/raw/{product}_{start}_{stop}.pkl').resample('5S').mean()
        df.columns = name
        data = pd.concat([data,df],axis=1)
        del df

def save_data_doublestar(data):
    columns_order = ['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']
    for c in columns_order :
    data[c] = medfilt(data[c],3)
    data.to_pickle(f'DATA/michotte/Datasets/DoubleStar/Datasets_DoubleStar/DS1_data_GSM_5S_{data.index[0].year}_{data.index[-1].year}.pkl')
    
    
def make_pos_doublestar(start,stop):
    pos = pd.read_pickle(f'/DATA/michotte/Datasets/DoubleStar/raw/ds1_xyz_gsm_{start}_{stop}.pkl')
    pos = pos[~pos.index.duplicated(keep='first')]
    pos.columns = ["X","Y","Z"]
    pos = pos.resample('5S').asfreq()
    pos = pos.interpolate(limit=int(12*5),limit_direction='both',center=True)
    pos['R'],pos['theta'],pos['phi'] = scc.cartesian_to_spherical(pos.X,pos.Y,pos.Z)
    pos.to_pickle(f'/DATA/michotte/Datasets/DoubleStar/Datasets_DoubleStar/DS1_pos_GSM_5S_{pos.index[0].year}_{pos.index[-1].year}.pkl')

def old_datasets_doublestar():
    data = pd.read_pickle(f'/DATA/nguyen/regions/doublestar.pkl')
    data['Tp'] = (data['Tpara']+2*data['Tperp'])/3 * 1e6
    data = data[['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']]
    data = data.resample('5S').mean()#.interpolate(limit=12,limit_area='inside',limit_direction='both')
    for x in data.columns:
        data[x] = medfilt(data[x], 3)
    data.to_pickle(f'data/5S_V2/data_TC_2004_2007.pkl')#A changer?


######## Fonction globale #############################################################################################################################################################################
#Ajouter entrée dates selon ce qu'on a rajouté avec le download_data

def make_datas(start,stop):
    
    make_dataset_doublestar(start,stop)
    make_dataset_themis(start,stop)
    make_dataset_MMS(start,stop)
    make_dataset_cluster(start,stop)







