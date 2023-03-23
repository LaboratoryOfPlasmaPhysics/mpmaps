import numpy as np
import pandas as pd
from scipy.signal import medfilt
from datetime import timedelta, datetime
import repere_and_coordinates as rac
        
def RemplaceToGSMCoordinates(data):
    if hasattr(data, 'Y_gsm' ) :      #hasattr check si une classe a bien la propriété reseignée en deuxième
        data['Y'] = data['Y_gsm']
        data['Z'] = data['Z_gsm']
    data = data[['X','Y','Z']]   
        
def load_data_themis(path_data ,path_pos,resample_time = None ):
    data = pd.read_pickle(path_data)
    data = data[~data.index.duplicated(keep='first')]
    data['Tp'] = (data['Tpara']+data['Tperp1']+data['Tperp2'])/3
    data = data[['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']]
    data[data.Tp<0] = np.nan
    data[data.Np>1000] = np.nan
    data['Tp']=data['Tp']*1.160451812e4
    data[abs(data.Vx)>10000] = np.nan
    data[abs(data.Vy)>10000] = np.nan
    data[abs(data.Vz)>10000] = np.nan
    if resample_time is not None :
        print(f'resampling to {resample_time}')
        data = data.resample(resample_time).mean()
    data = data.dropna()
    pos = pd.read_pickle(path_pos)
    pos[(pos.X==0) & (pos.Y==0) & (pos.Z==0)]=np.nan
    pos.dropna(inplace=True)
    data = data[data.index.isin(pos.index)]
    pos = pos[pos.index.isin(data.index)]
    pos['R'],pos['theta'],pos['phi'] = rac.cartesian_to_spherical(pos.X,pos.Y,pos.Z)    
    return data,pos

def load_data_and_pos_spacecraft(path_data, path_pos,resample_time=None,mode_sat=None):
    data = pd.read_pickle(path_data)
    pos = pd.read_pickle(path_pos)
    
    data = data[['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']]    
    data[data.Tp<=0] = np.nan
    data[data.Tp>=1e10] = np.nan
    data[data.Np>500] = np.nan    
    data[abs(data.Bx)>1000] = np.nan    
    data[abs(data.Vx)>2000] = np.nan
    data[abs(data.Vy)>2000] = np.nan
    data[abs(data.Vz)>2000] = np.nan
    
    
    if mode_sat is not None:
        status=pd.read_pickle(mode_sat)
        status=status.resample('5S').mean().interpolate(limit=13,limit_area='inside',limit_direction='both',method='polynomial', order=2)
        data=data[data.index.isin(status[status.values>=8].index)]
    
    if resample_time is not None :
        print(f'resampling to {resample_time}')
        data = data.resample(resample_time).mean()
        pos =  pos.resample(resample_time).mean()
       
 
    data = data[data.index.isin(pos.index)]

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    pos = pos[pos.index.isin(data.index)]    
    return data,pos


def load_data_spacecraft(data_name, pos_name, typefile='parquet',mode_sat=None,factor_Tp=None):
    data = pd.read_pickle(f'/DATA/nguyen/{data_name}')
    data['Tp'] = (data['Tpara']+2*data['Tperp'])/3
    data = data[['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp']]    
    data[data.Tp<0] = np.nan
    data[data.Np>1000] = np.nan    
    data[abs(data.Vx)>10000] = np.nan
    data[abs(data.Vy)>10000] = np.nan
    data[abs(data.Vz)>10000] = np.nan
    
    if factor_Tp is not None:
        data['Tp']=data.Tp*factor_Tp
    
    for x in data.columns:
        data[x] = medfilt(data[x], 3)
  
    if resample_time is not None :
        print(f'resampling to {resample_time}')
        data = data.resample(resample_time).mean()
    
    if typefile=='parquet':
        pos = pd.read_parquet(f'/DATA/nguyen/{pos_name}')
    else :
        pos = pd.read_pickle(f'/DATA/nguyen/{pos_name}')
        
    pos =  pos.resample('1T').mean().interpolate(method='linear').dropna()    
    RemplaceToGSMCoordinates(pos)        
    pos['R'],pos['theta'],pos['phi'] = rac.cartesian_to_spherical(pos.X,pos.Y,pos.Z)    
    data = data[data.index.isin(pos.index)]
    
    if mode_sat is not None:
        status=pd.read_pickle(mode_sat)
        status=status.resample('1T').mean().dropna()
        
        data=data[data.index.isin(status[status.values>=8].index)]

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()
    pos = pos[pos.index.isin(data.index)]    
    return data,pos


def load_OMNI_data(path,duration_mean_mins=0):
    omni_data = pd.read_pickle(path)
    omni_data=omni_data[datetime(2001,1,1):]
    omni_data = omni_data[['Bx', 'By', 'Bz', 'Np', 'Vx', 'Vy', 'Vz', 'Tp','Pd','Ma','Beta']]
    
    omni_data['V']=np.sqrt(omni_data['Vx'].values**2+omni_data['Vy'].values**2+omni_data['Vz'].values**2)
    omni_data['B']=np.sqrt(omni_data['Bx'].values**2+omni_data['By'].values**2+omni_data['Bz'].values**2)
    omni_data.loc[omni_data.B>9000]  =np.nan 
    omni_data.loc[omni_data.Np>900] =np.nan
    omni_data.loc[omni_data.Tp>9999900] =np.nan
    omni_data['CLA'] = np.sign(omni_data.By)*np.arccos(omni_data.Bz/np.sqrt(omni_data.By**2+omni_data.Bz**2))
    omni_data['COA'] = np.arctan(np.sqrt(omni_data.By**2+omni_data.Bz**2)/omni_data.Bx)
    omni_data.loc[omni_data.Bx<=1e-6].COA = np.pi/2
    #omni_data = omni_data.interpolate(limit = 15 ,limit_area='inside',limit_direction='both')
    if duration_mean_mins!=0:
        print(f'moyenne sur une durée de {duration_mean_mins} minutes')
        omni_data = omni_data.rolling(duration_mean_mins,center =True).mean().dropna()
    else :
        print('pas de moyenne temporelle')
    return omni_data


def clean_nightside_dipole(data, pos, omni_data, r_lim=None):
    def R_Shue1998(pos ,Pd, Bz ):
        r0 = (10.22+1.29*np.tanh(0.184*(Bz+8.14)))*Pd**(-1./6.6)
        a = (0.58-0.007*Bz)*(1+0.024*np.log(Pd))
        theta = pos.theta
        r = r0*(2./(1+np.cos(theta)))**a
        return r
    
    def make_Rav(theta,phi):
        a11 = 0.45 
        a22 = 1
        a33 = 0.8
        a12 = 0.18
        a14 = 46.6
        a24 = -2.2
        a34 = -0.6
        a44 = -618
        a = a11*np.cos(theta)**2 + np.sin(theta)**2 *( a22*np.cos(phi)**2 + a33*np.sin(phi)**2 ) +  a12*np.cos(theta)*np.sin(theta) * np.cos(phi)
        b = a14*np.cos(theta) +  np.sin(theta) *( a24*np.cos(phi) + a34*np.sin(phi) )
        c = a44
        delta = b**2 -4*a*c
        R = (-b + np.sqrt(delta))/(2*a)
        return R
    
    def R_Jerab05(pos, Np, V, Ma, B, gamma=2.15 ):
        C = 91.55
        D = 0.937*(0.846 + 0.042*B )
        R0 = make_Rav(0,0)
        theta = pos.theta
        Rav = make_Rav(theta,0)
        K = ((gamma-1)*Ma**2+2)/((gamma+1)*(Ma**2-1))
        R = (Rav/R0)*(C/(Np*V**2)**(1/6))*(1+ D*K)        
        return R

   
    pos = pos[pos.index.isin(data.index)]
 
    
    if r_lim is not None :
        Rmp = R_Shue1998(pos,omni_data[omni_data.index.isin(pos)].Pd, omni_data[omni_data.index.isin(pos)].Bz )
        
        pos = pos[pos.R>=(Rmp-r_lim[0])]
        Rbs = R_Jerab05(pos, omni_data[omni_data.index.isin(pos)].Np, omni_data[omni_data.index.isin(pos)].V, omni_data[omni_data.index.isin(pos)].Ma, omni_data[omni_data.index.isin(pos)].B, gamma=2.15 )
        pos = pos[pos.R<=(Rbs+r_lim[1])]
        data       = data[data.index.isin((pos.index))]
    return data,pos




def select_region_data(data_sat, pos_sat, omni_data, model_sat, region_nb, add_data_condition=None, r_lim=[5,5]):
    pred       =  pd.DataFrame(medfilt(model_sat.predict(data_sat),3))
    pred.index = data_sat.index
    pred      = pred.dropna()
    index_msh = pred[pred==region_nb]
    index_msh = index_msh.dropna()    
    data = data_sat.loc[data_sat.index.isin(index_msh.index)]

    data = pd.concat([data,pd.DataFrame({'V' : np.sqrt(data['Vx'].values**2+data['Vy'].values**2+data['Vz'].values**2) , 'B' :np.sqrt(data['Bx'].values**2+data['By'].values**2+data['Bz'].values**2), 'proba' : model_sat.predict_proba(data)[:,region_nb] }, \
                                        index=data.index)],axis=1)
    if add_data_condition is not None:
        data = add_data_condition(data)
    data = data.dropna()
    pos  = pos_sat.loc[ pos_sat.index.isin(index_msh.index)]     
    data, pos = clean_nightside_dipole(data.copy(), pos.copy(), omni_data.copy(), r_lim=r_lim)   
    #data['CLA'] = np.arctan2(omni.By,omni.Bz)
    #data['COA'] = np.arctan(np.sqrt(omni.By**2+omni.Bz**2)/omni.Bx)
    return data,pos

def OMNI_Safrankova(X_sat, omni, BS_standoff, dt=0,sampling_time='5S',vx_median =-406.2):  
    if dt != 0:
        vxmean = abs(omni.Vx.rolling(dt,center=True,min_periods=1).mean())
        #vxmean = abs(omni.Vx.rolling(int(2*dtm*timedelta(minutes=1)/(omni.index[-1]-omni.index[-2])),center=True,min_periods=1).mean())
    else:
        vxmean = abs(omni.Vx)
    BS_x0 = BS_standoff[BS_standoff.index.isin(X_sat.index)]
    BS_x0 = BS_x0.fillna(13.45)
    lag = np.array(np.round((BS_x0.values-X_sat.values)*6371/vx_median),dtype='timedelta64[s]')
    time = (X_sat.index-lag).round(sampling_time)   
    vx = pd.Series(name='Vx',dtype=float)
    vx  = vx.append(vxmean.loc[time],ignore_index=True).values
    vx = vx.fillna(abs(vx_mean))
    lag = np.array(np.round((BS_x0.values-X_sat.values)*6371/vx),dtype='timedelta64[s]')
    time = (X_sat.index-lag).round(sampling_time) 
    OMNI = pd.DataFrame(columns=omni.columns)    
    OMNI = OMNI.append(omni.loc[time], ignore_index=True)    
    OMNI.index = X_sat.index  
    return OMNI.dropna()


