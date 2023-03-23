import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import medfilt
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter as gf
from matplotlib.colors import LogNorm, SymLogNorm, Normalize
from sklearn.metrics import r2_score, mean_squared_error
import sys
sys.path.append('../space/')
from space.models import planetary as smp
from space.coordinates import coordinates as scc

def scores(ytrue,ypredict,wanted_scores = 'all'):
    if wanted_scores == 'RMSE' or wanted_scores == 'rmse':
        return np.sqrt(mean_squared_error(ytrue,ypredict))
    elif wanted_scores == 'R2' or wanted_scores == 'r2':
        return r2_score(ytrue,ypredict)
    else :
        return np.sqrt(mean_squared_error(ytrue,ypredict)),r2_score(ytrue,ypredict)

def histedges_n_pts_equal(x, nbin):
    bins = np.interp(np.linspace(0, len(x), nbin + 1), np.arange(len(x)),np.sort(x))
    if len(bins)==len(np.unique(bins)):
        return  bins
    else:
        val,ct = np.unique(bins, return_counts=True)
        for x in val[ct>1]:
            idx = np.where(bins==x)[0]
            if idx[-1]!=len(bins)-1:
                dx  = (bins[idx[-1]+1]-bins[idx[-1]])/len(idx)
                bins[idx] = bins[idx]+np.array([i*dx for i in range(len(idx))])
            else :
                dx=(bins[idx[0]-1]-bins[idx[0]])/len(idx)
                bins[idx] = bins[idx]+np.array([i*dx for i in np.flip(np.arange(len(idx)))])
        return bins
    
def histo1D(x, y, bins, **kwargs):
    X = stats.binned_statistic(x,x, statistic=kwargs.get('statistic','mean'), bins=bins)[0]
    Y = stats.binned_statistic(x,y, statistic=kwargs.get('statistic','mean'), bins=bins)[0]
    count = stats.binned_statistic(x,y, statistic='count', bins=bins)[0]
    c =count.copy()
    c[c==0]=1
    SEM_x = np.array(stats.binned_statistic(x,x, statistic='std', bins=bins)[0])/np.sqrt(c)
    SEM_y = np.array(stats.binned_statistic(x,y, statistic='std', bins=bins)[0])/np.sqrt(c)
    
    if kwargs.get('medfilt',False):
        X = medfilt(X,kwargs.get('medfilt_value',3))
        Y = medfilt(Y,kwargs.get('medfilt_value',3))
        SEM_x = medfilt(SEM_x,kwargs.get('medfilt_value',3))
        SEM_y = medfilt(SEM_y,kwargs.get('medfilt_value',3))
        
    return X,Y,SEM_x,SEM_y,count


def hist_2d(x, y, qty_x, qty_y, qty_z, **kwargs):
    statistic = kwargs.get('statistic','mean')
    value = stats.binned_statistic_2d(qty_x, qty_y, qty_z, statistic=statistic, bins=[x,y]).statistic
    if kwargs.get('gaussian',False) :
        value =filter_nan_gaussian_conserving2(value, sigma=kwargs.get('sigma',(1,1)))
    return value

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



    
def make_least_squares(f,a,x,y,x0,**kwargs):
    def func(a,x,y):
        return f(a,x)-y
    res_lsq = least_squares(func, x0=x0, args=(x, y) ,**kwargs)
    return res_lsq.x,np.sqrt(np.diagonal(np.linalg.inv(res_lsq.jac.T.dot(res_lsq.jac))))
 
    
def plot_hist1D(x,y,sem_x,sem_y, **kwargs):
    if 'fig_element' not in kwargs:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize',(8,5)))
        ax.set_xlabel(kwargs.get('x_label',None))
        ax.set_ylabel(kwargs.get('y_label',None))
        fig.suptitle(kwargs.get('title',None))
    
    else :
        fig, ax = kwargs['fig_element']
    if kwargs.get('with_line',False):
        ax.plot(x,y,kwargs.get('style_line','-'),color=kwargs.get('color','k'),lw=kwargs.get('lw',1.),\
            label=kwargs.get('label',None))
        ax.fill_between(x,y-sem_y,y+sem_y,color=kwargs.get('color','k'),alpha = kwargs.get('alpha',0.4))
    else :
        ax.errorbar(x,y,yerr=sem_y,xerr=sem_x,ls=kwargs.get('style_line','none'),marker=kwargs.get('marker','D'),markerfacecolor=kwargs.get('markerfacecolor','none'), color=kwargs.get('color','k'),label=kwargs.get('label',None))
        if 'labels_pts' in kwargs :
            dup = kwargs.copy()
            dup.pop('color')
            ax =  annotation_labels(ax,kwargs['labels_pts'], x, y, yerr=sem_y, color=kwargs.get('color_annot',kwargs.get('color','k')), **dup)
    ax.set_xlim(kwargs.get('x_lim',None))
    ax.set_ylim(kwargs.get('y_lim',None))
    if 'label' in kwargs :
        ax.legend(loc=kwargs.get('loc_legend','best'))
    return fig,ax

def annotation_labels(ax,labels, x, y, **kwargs):
    yerr = kwargs.get('yerr',np.zeros_like(y))
    text_offset =kwargs.get('text_offset',(0,2))
    for i in range(len(labels)):
        ax.annotate(labels[i], (x[i],y[i]+yerr[i]), textcoords="offset points", xytext=text_offset, ha='center',alpha=kwargs.get('alpha',0.6),color=kwargs['color'])
    return ax

def select_data_with_condition(data,cond):
    if isinstance(data, list):
        return [d[cond] for d in data]
    else :
        return data[cond]
    


def select_data_in_slice(list_to_slice,slice_pos,**kwargs):
    value_slice = kwargs.get('value_slice',0) 
    thickness   = kwargs.get('thickness',1) 
    s_list = select_data_with_condition(list_to_slice,abs(slice_pos-value_slice)<=thickness)
    return s_list
    
def make_bins_and_meshgrid(**kwargs):
    x_lim = kwargs.get('x_lim',(-20,15))
    y_lim = kwargs.get('y_lim',(-30,30))
    n_xbins = kwargs.get('n_xbins',100)
    n_ybins = kwargs.get('n_ybins',100)
    x = np.linspace(x_lim[0],x_lim[1],n_xbins+1)
    y = np.linspace(y_lim[0],y_lim[1],n_ybins+1)
    
    X, Y = np.meshgrid(0.5*(x[1:]+x[:-1]), 0.5*(y[1:]+y[:-1]),indexing=kwargs.get('indexing','ij'))
    return x,y,X,Y

def make_hist2d(qty, pos_abs, pos_ord, pos_slice,**kwargs):
    s_qty, s_pos_abs, s_pos_ord  = select_data_in_slice([qty,pos_abs, pos_ord],pos_slice,**kwargs)
    x,y,X,Y = make_bins_and_meshgrid(**kwargs)
    values = hist_2d(x, y, s_pos_abs, s_pos_ord, s_qty, **kwargs)
    return x,y,X,Y,values

def plot_hist2d(qty, pos_abs, pos_ord, pos_slice, fig, ax,**kwargs):
    s_qty, s_pos_abs, s_pos_ord  = select_data_in_slice([qty,pos_abs, pos_ord],pos_slice,**kwargs)
    x,y,X,Y = make_bins_and_meshgrid(**kwargs)
    values = hist_2d(x, y, s_pos_abs, s_pos_ord, s_qty, **kwargs)
    hist=ax.pcolormesh(X, Y,  values, cmap=kwargs.get('cmap','jet'), norm = kwargs.get('norm',Normalize(np.min(values),np.max(values))) )
    cbar = kwargs.get('cbar','ax')
    if cbar=='return':
        return hist
    elif cbar=='ax':
        fig.colorbar(hist, ax=ax)
    

def plot_streamlines(qty_abs, qty_ord, pos_abs, pos_ord, pos_slice, fig, ax,**kwargs):
    s_qty_abs, s_qty_ord, s_pos_abs, s_pos_ord  = select_data_in_slice([qty_abs, qty_ord, pos_abs, pos_ord],pos_slice,**kwargs)

    kwargs['indexing']='xy'
    x,y,X,Y = make_bins_and_meshgrid(**kwargs)
    values_abs = hist_2d(x, y, s_pos_abs, s_pos_ord, s_qty_abs, **kwargs)
    values_ord = hist_2d(x, y, s_pos_abs, s_pos_ord, s_qty_ord, **kwargs)
    #X = (X[1:,1:]+X[:-1,:-1])*0.5
    #Y = (Y[1:,1:]+Y[:-1,:-1])*0.5
    if kwargs.get('imf',None) is not None:
        statistic = kwargs.get('statistic','mean')
        if statistic=='mean':
            imf_abs = np.mean(kwargs['imf'][0])
            imf_ord = np.mean(kwargs['imf'][1])
        elif statistic=='median':
            imf_abs = np.median(kwargs['imf'][0])
            imf_ord = np.median(kwargs['imf'][1])
            
        plane  = kwargs.get('plane','XY')
        if plane == 'XY':
            r,theta,phi = scc.cartesian_to_spherical(X,Y,kwargs.get('slice',0))
        elif plane == 'XZ':
            r,theta,phi = scc.cartesian_to_spherical(X,kwargs.get('slice',0),Y)
        elif plane == 'YZ':
            r,theta,phi = scc.cartesian_to_spherical(kwargs.get('slice',0),X,Y)
        if 'bs' not in kwargs:
            bs = smp.bs_jelinek2012
        if 'mp' not in kwargs:
            mp = smp.mp_shue1998
        
        if plane !='YZ':
            print('b')
            r_bs = bs(theta,phi,coord_sys='spherical')[0]
            values_abs[np.isnan(values_abs) & (r>=r_bs.T)]=imf_abs
            values_ord[np.isnan(values_ord) & (r>=r_bs.T)]=imf_ord
            values_abs[np.isnan(values_abs)]=0
            values_ord[np.isnan(values_ord)]=0
        else :
            print("a")
            r_mp = mp(theta,phi,coord_sys='spherical')[0]
            values_abs[(np.isnan(values_abs) | (values_abs==0)) & (r>r_mp.T)]=imf_abs
            values_ord[(np.isnan(values_ord) | (values_ord==0)) & (r>r_mp.T)]=imf_ord
            
        
    else :
        values_abs[np.isnan(values_abs)]=0
        values_ord[np.isnan(values_ord)]=0
    ax.streamplot(X,Y,values_abs.T,values_ord.T,start_points = kwargs.get('start_points',None) ,density=kwargs.get('density',1),color=kwargs.get('color','k'),linewidth=kwargs.get('linewidth',0.7),arrowsize=kwargs.get('arrowsize',0.7), maxlength=kwargs.get('maxlength',4.))