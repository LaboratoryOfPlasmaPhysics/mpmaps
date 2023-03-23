import numpy as np
import pandas as pd
import datetime
import sys
from glob import glob
import re
import os
from astropy.constants import R_earth
import speasy as spz

from functools import partial
from multiprocessing import Pool 
import spacepy.pycdf as pycdf
from datetime import tzinfo, timedelta, datetime
sys.path.append('./space')
sys.path.append('Python_functions')
sys.path.append('./')
import utilities as uti


#Chemins à changer pour récupérer datas
#spwc à remplacer
############################################### Themis ##################################################################################################################################
def download_data_themis(startdate,stopdate):
    
    dens = pd.read_pickle(f'/DATA/michotte/Datasets/THEMIS/raw/THB_L2_MOM_thb_peim_density_{startdate}_{stopdate}.pkl')
    start = startdate #datetime(2021,7,1,0)
    stop = stopdate #datetime(2022,6,28,0) 
    dt=timedelta(days=7)
    N= int((stop-start)/dt)
    
    vv = [spz.get_data(f"cdaweb/THA_L2_ESA/tha_peif_t3Q", start+i*dt, start+(i+1)*dt ) for i in range(N)]
    pd.concat([v.to_dataframe(True) for v in vv]).dropna()
    
    for sat in ["A","B","C","D","E"]:
        f(sat,start,stop)
    with Pool(5) as p:
        p.map(f(sat,start,stop),["D","E"]) 
    download_spectros(start,stop)
    
############################################# DoubleStar #############################################################################################################################

def download_data_doublestar(startdate,stopdate):
    start_stop_dt = [startdate,stopdate,timedelta(days=7)]
    products  =  ["ds1_n", "ds1_v_gsm", "ds1_t", "ds1_b_gsm", "ds1_xyz_gsm" ]
    intervals = intervals(start_stop_dt)
    import_data(products, start_stop_dt,intervals,'DoubleStar')
    
################################################# Cluster ###############################################################################################################################
def download_cluster_data(startdate,stopdate):
    #C1
    start_stop_dt =[startdate,stopdate,timedelta(days=7)]
    products  =  ["c1_hia_dens", "c1_hia_v_gsm", "c1_hia_tpar" ,"c1_hia_tperp", "c1_b_gsm", "c1_xyz_gsm" ]
    intervals = intervals(start_stop_dt)
    
    import_data(products, start_stop_dt,intervals,'Cluster')
    
    #C3
    start_stop_dt =[startdate,stopdate,timedelta(days=7)]
    products  =  ["c3_hia_dens", "c3_hia_v_gsm", "c3_hia_tpar" ,"c3_hia_tperp", "c3_b_gsm", "c3_xyz_gsm" ]
    intervals = intervals(start_stop_dt)

    import_data(products, start_stop_dt,intervals,'Cluster')
    
######################################### MMS ##########################################################################################################################################

def download_MMS_data(startdate,stopdate):
    #Phase 1
    L=[1,2,3]
    start_stop_dt = [startdate,stopdate,timedelta(days=1)]
    products  =  ["mms1_dis_ni", "mms1_dis_vgse", "mms1_dis_tpara" ,"mms1_dis_tperp" ]

    intervals=intervals(start_stop_dt)

    import_data(products,start_stop_dt,intervals,'MMS')

    #Phase 2

    start_stop_dt = [startdate,stopdate,timedelta(hours=6)]
    products=["mms1_b_gsm"]

    intervals=intervals(start_stop_dt)

    import_data(products,start_stop_dt,intervals,'MMS')

    #Phase 3
    start_stop_dt = [startdate,stopdate,timedelta(hours=6)]
    products=["mms1_b_gsm"]

    intervals=intervals(start_stop_dt)

    import_data(products,start_stop_dt,intervals,'MMS')

    #Phase 4

    start_stop_dt = [startdate,stopdate,dt==timedelta(days=1)]
    intervals=intervals(start_stop_dt,dt)

    v =[spwc.get_orbit('mms1',interval[0] , interval[1],  coordinate_system='gsm') for interval in intervals]
    v=spwc.common.variable.merge(v).to_dataframe(True)
    v.to_pickle(f'/DATA/michotte/Datasets/MMS/raw/mms1_orbit_gsm_{start_stop_dt[0].year}_{start_stop_dt[1].year}.pkl')
    del v


    #Phase 5
    start_stop_dt = [startdate,stopdate,timedelta(days=1)]
    intervals=intervals(start_stop_dt)

    v =[spz.get_data(f'cdaweb/MMS1_FGM_SRVY_L2/mms1_fgm_r_gsm_srvy_l2',interval[0] , interval[1]) for interval in intervals]
    v=spwc.common.variable.merge(v).to_dataframe(True)
    v.to_pickle(f'/DATA/michotte/Datasets/MMS/raw/mms1_fgm_r_gsm_srvy_l2_{start_stop_dt[0].year}_{start_stop_dt[1].year}.pkl')
    del v


########### fonction globale #########################################################################################################

def download_data(startdate,stopdate):
    
    download_data_themis(startdate,stopdate)
    download_cluster_data(startdate,stopdate)
    download_data_doublestar(startdate,stopdate)
    download_MMS_data(startdate,stopdate)
    
#####utilities#########################################################################################################

def f(sat,start,stop):
    
    dt=timedelta(days=7)
   
    N= int((stop-start)/dt)
    
    wanted_data = {       
        f'TH{sat}_L2_FGM' : [f'th{sat.lower()}_fgs_gsmQ'],
        f'TH{sat}_L2_MOM':  [f'th{sat.lower()}_peim_t3_mag', 
                         f'th{sat.lower()}_peim_velocity_gsm', f'th{sat.lower()}_peim_density'],
        f'TH{sat}_OR_SSC' : ['XYZ_GSM']}
    
    
    for k in wanted_data.keys():
        for p in wanted_data[k]:
            print(p)
            vvv =[spz.get_data(f"cdaweb/{k}/{p}", start+i*dt, start+(i+1)*dt ) for i in range(N)]
            v= pd.concat([vv.to_dataframe(True) for vv in vvv])
            v.to_pickle(f"/DATA/michotte/Datasets/THEMIS/raw/{k}_{p}_{start.year}_{stop.year}.pkl")
            del v

def intervals(start_stop_dt):
    
    start=start_stop_dt[0]
    stop=start_stop_dt[1]
    dt=start_stop_dt[2]
    N= int((stop-start)/dt)
    
    intervals = [(start+i*dt, start+(i+1)*dt ) for i in range(N)]
    return(N,str(start.year),start + N*dt)

def import_data(products, start_stop_dt,intervals,name_sat):
    
    start=start_stop_dt[0]
    stop=start_stop_dt[1]
    
    for product in products:
        v =[spwc.get_data(f'amda/{product}',interval[0] , interval[1]) for interval in intervals]
        v=spwc.common.variable.merge(v).to_dataframe(True)
        v.to_pickle(f'/DATA/michotte/Datasets/{name_sat}/raw/{product}_{start.year}_{stop.year}.pkl')
        del v
        
def download_spectros(start,stop):
    
    v =spz.get_data(f"cdaweb/{'THA_L2_ESA'}/{'tha_peir_en_eflux'}", start, stop ) 
    with Pool(5) as pool:
        pool.map(f(sat,start,stop), ["A","B","C","D","E"])
    pool.close()

