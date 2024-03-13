################################################
# est_param_with_tides.py
# Tomoya Takano (ERI)
# November 2020
################################################

import sys
import os 
import glob
import time
import copy
import pickle
import h5py as h5 
import numpy as np
import random
import scipy.io as io
import math
from obspy import UTCDateTime as UTC

import gc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from obspy.core import UTCDateTime, Stream
import scipy.signal as signal

from scipy import interpolate

from obspy.geodetics import gps2dist_azimuth as gps2dist
from obspy.geodetics import kilometer2degrees
from obspy.geodetics import degrees2kilometers
from obspy.geodetics import locations2degrees
from KalmanFilter_tides import Klf
from scipy import optimize


from obspy.signal.detrend import polynomial
from obspy.core.inventory.inventory import read_inventory
from obspy import read

import lang 
import dd

try : 
    import ipdb 
except :
    pass 






def main_dvv(input_user={}):
    in_ = {}
    in_['path']         = 'C1_04_xcorr_test__xcorr_norm/'
    in_['cc_cmp']       = ['UU'] #...
    in_['cc_maxlag']    = 500 #[s]
    in_['overlap_perc'] = 50.#[%]
    in_["start_time"]   = 0#[s]
    in_["end_time"]     = 20#[s]
    in_['format']       = 'float16' #...
    in_['p']            = [1, 100]#[s] periods
    in_                 = lang.parse_options(in_,input_user)

    list_file     = glob.glob(in_['path']+'/*h5')
    
    start         = time.process_time()
    compos        = in_['cc_cmp']
    random.shuffle(list_file)
    h5file_tide   = h5.File("tides_amp_phase_japan_2012-2018.h5",'r')
    
    for file_in in list_file:
        outdir = os.path.dirname(file_in) + '/2012-2015_m2_kalman_explanatory_p'+str(in_['p'][0])+'-'+str(in_['p'][1])+"s_lag"+str(in_['start_time'])+'-'+str(in_['end_time'])+'s_' + str(len(compos)) + 'compos/'
        
        if not os.path.exists(outdir):
            mkdir(outdir)

        file_out  = outdir + os.path.basename(file_in[:-3]) + '_karman.h5'
        lock_file = outdir + os.path.basename(file_in[:-3]) + '_karman.lock'

        if os.path.isfile(lock_file) or os.path.isfile(file_out):
            dd.dispc(file_out + " and/or lock_file already exist",'r','n')
            continue
        create_lock_file(lock_file)
        
        fin  = h5.File(file_in, "r")
        fout = h5.File(file_out, "a")
        if '/md' in fin: fout.copy(fin['/md'],'/md')
        if '/md_c' in fin: fout.copy(fin['/md_c'],'/md_c')
        if '/in_' in fin: fout.copy(fin['/in_'],'/in_')
        if '/ref_nstack' in fin: fout.copy(fin['/ref_nstack'],'/ref_nstack') # not correct after rotation ...
        if '/cc_nstack' in fin: fout.copy(fin['/cc_nstack'],'/cc_nstack') # not correct after rotation ...
        add_metadata(fout,'in_stack',in_)

        len_corr    = len(fin['md_c/t'][:])
        tau         = fin['md_c/tau'][()]
        
        
        for ind, sta in enumerate(fin['md/id'][:]):
            group_ref_id = '/ref/'+sta[0].decode('utf8')+'/'+sta[1].decode('utf8')
            group_cc_id  = '/cc/'+ sta[0].decode('utf8') +'/'+sta[1].decode('utf8')
            group_id     = sta[0].decode('utf8')+'/'+sta[1].decode('utf8')
            aa       = str(sta[0].decode('utf8')).split('.')
            h5_node  = "/Hi-net/N." + str(aa[2])
            tide_amp = h5file_tide[h5_node+"/amp"][()]
            tide_ph  = h5file_tide[h5_node+"/phase"][()]
            dvv_t, std_t, beta, model, dAIC, h0 = calc_dvv(fin, tide_amp, tide_ph, in_, len_corr, compos, group_cc_id, group_ref_id, tau)
            
            fout.create_dataset('dvv',   data = dvv_t[:], dtype=in_['format'])
            fout.create_dataset('std',   data = std_t[:], dtype=in_['format'])
            fout.create_dataset('beta',  data = beta, dtype=in_['format'])
            fout.create_dataset('model', data = model, dtype=in_['format'])
            fout.create_dataset('dAIC',  data = dAIC, dtype=in_['format'])
            fout.create_dataset('h0',    data = h0, dtype=in_['format'])
            dd.dispc(group_cc_id + " Time : " + str(time.process_time() - start),'g','n')
            del dvv_t
            del std_t
            gc.collect()

        fout.close()
        fin.close()
        os.remove(lock_file)


def calc_dvv(fin,tide_amp, tide_ph, in_, len_corr, compos, group, group_ref, tau):
    nb_date  = len(fin['md_c/date1'][365*24*4:365*4*24+365*3*24])
    nb_compo = len(compos)
    data     = np.zeros((nb_date, len_corr, len(compos)), dtype=in_['format'])
    data_ref = np.zeros((len_corr, len(compos)), dtype=in_['format'])

    #q_ref    = np.array([1E-6, 1E-6, 1E-4, 1, 1E-4, 1, 1E-4, 1, 1E-3])
    q_ref    = np.array([1E-7, 1E-12, 1E-4, 1.,  1E-3])
    q_ref2   = np.array([1E-7, 1E-12,  1E-3 ])
    fe       = 1/tau
    m0       = int(np.floor(len(fin['md_c/t'][:])*0.5))

    for ch in compos:
        if ch in fin[group]:
            data[:,:,compos.index(ch)]   =fin[group][ch][365*24*4:365*4*24+365*3*24,:].astype(in_['format'])
            #data_ref[:,compos.index(ch)] =fin[group_ref][ch][:].astype(in_['format'])

    time     = fin['md_c/t'][:][m0:m0+int(in_["cc_maxlag"]*fe)]
    date_vec = np.arange(0,len(fin['md_c/date1'][365*24*4:365*4*24+365*3*24]),1)
    ccfs     = np.zeros((len(compos), nb_date, len(time)), dtype=in_['format'])
    tccfs     = np.zeros((len(compos), nb_date, len(time)), dtype=in_['format'])
    ccfr     = np.zeros((len(compos),len(time)), dtype=in_['format'])
    beta     = np.zeros(len(ccfs[0])+1)
    
    win      = np.zeros(len(time))        
    ts       = int(in_['start_time']/tau)
    te       = int(in_['end_time']/tau)
    win2     = signal.tukey(te-ts, 0.1)
    
    for i in range(ts, te): win[i] = win2[i-ts]

    for icmp in range(nb_compo):
        for idates in range(nb_date):
            khour  = UTC(fin['md_c/date1'][idates]).hour
            if khour == 9:
                trace = np.zeros((len(time)))
            else:
                trace = data[idates,m0:m0+int(in_["cc_maxlag"]*fe),icmp]#int(in_["cc_maxlag"]*fe-1),icmp]    
            ccfs[icmp, idates, :] = trace*win
            del trace
            gc.collect()
            #tccfs[icmp, idates, :] = trace
    del data
    del data_ref
    gc.collect()

    h0, ccf_ref = est_h0(ccfs, ccfr, nb_compo,ts, te, nb_date)
    try:
        alpha_t, Vt, lnL, mask_param = eKlf(nb_compo, ccfs, ccf_ref, tau, ts, te, beta, h0,  [q_ref[0], 1E-10])
    except:
        return np.zeros((nb_date)), np.zeros((nb_date)),np.zeros((5)),0,0,0
    
    rawdvv   = alpha_t[:,1]
    rawamp   = alpha_t[:,0]
    ccf_ref2 =  cal_ref(ccfs,rawdvv,rawamp,tau)
    h0 = est_h0_2(ccfs,ccf_ref2, nb_compo, ts,te,nb_date)

    q_init   = [1.,1.,1.,1.,1.]
    lnL_tmp  = lnL

    bounds   = [[1E-3, 1E3], [0.2,100], [1E-1, 1E2], [-180,180], [-10,10]]#[[5E-2, 2E1], [1E-3, 2E1], [0.1, 10.], [0.1,60], [0.1, 10.], [0.1,60], [0.1, 10.], [0.1,60], [0.1,10]]
    
    q_est, min, log = optimize.fmin_l_bfgs_b(lambda q_tmp: likelihood([q_tmp[0], q_tmp[1], q_tmp[2], q_tmp[3],q_tmp[4]],q_ref, date_vec,nb_compo,ccfs, ccf_ref, h0, tau, ts, te, tide_amp,tide_ph, ref=lnL), q_init, epsilon=1e-5,bounds=bounds, approx_grad = True)
    alpha_t, Vt, lnL, beta, mask_param = likelihood(q_est, q_ref,date_vec,nb_compo,ccfs, ccf_ref, h0, tau, ts, te, tide_amp, tide_ph, flag=1)

    qtmp     = q_est#*q_ref
    q_init2  = [1.,1.,1.]
    bounds   = [[1E-3, 1E3],[0.2,100],  [-10,10]]

    q_est2, min2, log2 = optimize.fmin_l_bfgs_b(lambda q_tmp: likelihood2([q_tmp[0], q_tmp[1],q_tmp[2]], q_ref2, date_vec, nb_compo,ccfs, ccf_ref, h0, tau, ts, te, ref=lnL_tmp), q_init2, epsilon=1e-5, bounds=bounds, approx_grad = True)
    alpha_t2, Vt2, lnL2, beta2, mask_param2 = likelihood2(q_est2, q_ref2,date_vec,nb_compo,ccfs, ccf_ref, h0, tau, ts, te, flag=1)
    
    dvv  = alpha_t[:,1]*1E2
    err  = np.sqrt(Vt[:,1,1])*1E2
    dlnL = lnL2 - lnL
    dAIC = 2.*(dlnL + 2.)
    
    return dvv, err, q_est*q_ref, beta, dAIC, h0

def likelihood(qm, q_ref,date_vec,nb_cmp,ccfs1, ccf_r, h0, delta, ts, te, tide_amp, tide_ph, flag=0, ref=0):
    # q0, d_am, d_ph, se_am, se_ph, m2_am, m2_ph, d1
    # np.array([1E-6, 1E-4, 1, 1E-4, 1, 1E-4, 1, 1E-3])
    q0              = qm * q_ref
    D2R             = np.pi/180.

    diurnal_deg     = 1.*14.999999999992225
    yeardeg         = 1.*14.999999999992225/365.
    semidiurnal_deg = 2.*14.999999999992225
    tides_deg       = 1.9322736*14.999999999992225

    St = q0[2] * np.cos((date_vec[:] * tides_deg + tide_ph + q0[3])*D2R) #+ q0[4] *i np.cos((date_vec[:] * yeardeg + q0[5])*D2R)

    alpha_t, Vt, lnL, mask_param = eKlf(nb_cmp,ccfs1, ccf_r, delta, ts, te, St, h0, [q0[0], q0[1]], Pt=np.array([[q0[0],0], [0,q0[1]]]), at0 = [1, q0[4]])

    
    if flag == 0:
        np.set_printoptions(precision=2,linewidth=200)
        np.set_printoptions(precision=8)
        return(-lnL+ref)
        
    if flag == 1:
        return (alpha_t, Vt, lnL, St, mask_param)

def likelihood2(qm, q_ref2,date_vec,nb_cmp,ccfs1, ccf_r, h0, delta, ts, te, flag=0, ref=0):
    q0        = qm * q_ref2
    D2R       = np.pi/180.

    ##Diurnal, Semi-diurnal, and M2 calculation
    diurnal_deg     = 1.*14.999999999992225
    semidiurnal_deg = 2.*14.999999999992225
    yeardeg         = 1.*14.999999999992225/365.
    
    #St  = q0[2] * np.cos((date_vec[:] * yeardeg + q0[3])*D2R)
    #St = q0[1] * np.cos((date_vec[:] * diurnal_deg + q0[2])*D2R) + q0[3] * np.cos((date_vec[:] * semidiurnal_deg + q0[4])*D2R)
    St = np.zeros(len(ccfs1[0])+1)

    alpha_t, Vt, lnL, mask_param = eKlf(nb_cmp,ccfs1, ccf_r, delta, ts, te, St, h0, [q0[0], q0[1]], Pt=np.array([[q0[0],0], [0,q0[1]]]), at0 = [1, q0[2]])
        
    if flag == 0:
        np.set_printoptions(precision=2,linewidth=200)
        np.set_printoptions(precision=8)
        return(-lnL+ref)
        
    if flag == 1:
        return (alpha_t, Vt, lnL, St, mask_param)

def cal_ref(ccfs_all,at,A,delta):
    ccf_r2 = np.zeros([ccfs_all.shape[0],ccfs_all.shape[2]])
    lag_time = (np.arange(ccfs_all.shape[2]))*delta

    for itime in range(ccfs_all.shape[1]):
        α0 = 0
        α0 = at[itime]#/at[itime][0]
        for cmp1 in range(9):
            y_tmp = ccfs_all[cmp1][itime]
            lag_time2 = (np.arange(ccfs_all.shape[2]))*delta*(1+α0)
            f2 = interpolate.interp1d(lag_time, y_tmp, kind="quadratic", fill_value='extrapolate')
            ccf_r2[cmp1] += f2(lag_time2)
    for cmp1 in range(9):
        ccf_r2[cmp1]  /= ccfs_all.shape[1]
    return ccf_r2

def cal_semidiurnal( at, ph, mask_param, date_vec):
    D2R      = np.pi/180.
    semidiurnal_deg = 2.*14.999999999992225
    X = 1E-3*np.cos((date_vec[:] * semidiurnal_deg + ph)*D2R)
    y = at[:,1].copy()
    y[np.isnan(y)] = 0.

    ccc  = y @ X
    beta = ccc / (X@X)
    res  = -2 * ccc * beta + beta * beta *X@X + y@y

    Sdi  = beta * X
    amp  = (np.max(Sdi) - np.min(Sdi))/2.

    res = ((y-Sdi)**2).mean()
    return Sdi, y, res, amp, beta

def cal_diurnal( at, ph, mask_param, date_vec):
    D2R         = np.pi/180.
    diurnal_deg = 1.*14.999999999992225
        
    X = 1E-3*np.cos((date_vec[:] * diurnal_deg + ph)*D2R)
    y = at[:,1].copy()
    y[np.isnan(y)] = 0.

    ccc  = y @ X
    beta = ccc / (X@X)
    res  = -2 * ccc * beta + beta * beta *X@X + y@y

    Sdi  = beta * X
    amp  = (np.max(Sdi) - np.min(Sdi))/2.

    res = ((y-Sdi)**2).mean()
    return Sdi, y, res, amp, beta

def cal_tides( at, ph, tide_amp,tide_ph, mask_param, date_vec):
        
    D2R       = np.pi/180.
    tides_deg = 1.9322736*14.999999999992225
    X = np.zeros((len(date_vec), 1))
    X = 1E+3*tide_amp * 0.5 * np.cos((tides_deg * date_vec[:] + tide_ph + ph)*D2R )
        
    #observation vector y
    y = at[:,1].copy()
    y[np.isnan(y)] = 0
        
    ccc  = y @ X
    beta = ccc / (X@X)
    res  = -2 * ccc * beta + beta * beta *X@X + y@y
        
    #modeled tides
    St = beta * X
            
    amp = (np.max(St) - np.min(St))/2.
    res = ((y-St)**2).mean()
    
    return St, y, res, amp, beta

def est_h0(ccfs1, ccf_r, nb_cmp,ts,te,ndays):
    h0 = 0.
    mask0 = [True]*(ccfs1.shape[2])
    #for i in range(ts,te): mask0[i] = True
    for icmp in range(nb_cmp):
        ccfs        = ccfs1[icmp]
        mccfs       = np.ma.masked_array(ccfs,np.isnan(ccfs))
        ccf_r[icmp] = mccfs.mean(axis=0)
        scale       = ccfs[:,mask0].dot(ccf_r[icmp][mask0])/ccf_r[icmp][mask0].dot(ccf_r[icmp][mask0])
        mask        = (scale>0.5) & (scale<5.)
        h0         += np.mean(((mccfs[mask])[:,mask0]- ccf_r[icmp][mask0])**2)
        mccfs[mask] = np.diag(1./scale[mask]) @ np.array(mccfs[mask])
        ccf_r[icmp] = mccfs[mask].mean(axis=0)

    h0 /= nb_cmp
    return(h0, ccf_r)

def est_h0_2(ccfs1, ccf_r, nb_cmp,ts,te,ndays):
    h0 = 0.
    mask0 = [False]*(ccfs1.shape[2])
    for i in range(ts,te): mask0[i] = True
    for icmp in range(nb_cmp):
        ccfs        = ccfs1[icmp]
        mccfs       = np.ma.masked_array(ccfs,np.isnan(ccfs))
        #ccf_r[icmp] = mccfs.mean(axis=0)
        scale       = ccfs[:,mask0].dot(ccf_r[icmp][mask0])/ccf_r[icmp][mask0].dot(ccf_r[icmp][mask0])
        mask        = (scale>0.5) & (scale<2.)
        h0         += np.mean(((mccfs[mask])[:,mask0]- ccf_r[icmp][mask0])**2)
        mccfs       = np.diag(1./scale[mask]) @ np.array(mccfs[mask])
        #ccf_r[icmp] = mccfs[mask].mean(axis=0)

    h0 /= nb_cmp    
    return h0

def eKlf(nb_cmp,ccfs_all,ccf_r,delta,ts,te,β,h0,Q0,Pt=np.array([[1E-5,0.],[0.,1E-8]]),at0=np.array([1.,0.])):

    mask0 = [False]*(ccfs_all.shape[2])
    for i in range(ts, te): mask0[i] = True
    
    lag_time = (np.arange(ccfs_all.shape[2])*delta)[mask0]
    lnL = 0
    att = np.zeros([len(ccfs_all[0]),2])
    αt = np.zeros([len(ccfs_all[0]),2])
    Ptt = np.zeros([len(ccfs_all[0]),2,2])
    Vt = np.zeros([len(ccfs_all[0]),2,2])
    Qt = np.zeros([2,2])
    mask_param = np.full([len(ccfs_all[0]),nb_cmp], False, dtype=np.bool) 
        
        
    itime = 0
    at = at0.copy() #Initial value
    Qt = np.array([[Q0[0],0],[0,Q0[1]]]) #Pt*ϵ #[[1E-6,0],[0,1E-9]] #Initial value
    Tayler_Series = np.zeros([nb_cmp, 5, (te-ts)*1]) # Up to 5th order
    ms_r = np.zeros([nb_cmp])

    for icmp in range(nb_cmp):
        Tayler_Series[icmp] = [deri(ccf_r[icmp], i, delta)[mask0] for i in range(5)]
        ms_r[icmp]          = np.sum((ccf_r[icmp][mask0]**2))
            
            
    for iday in range(ccfs_all.shape[1]):            
        yt = ccfs_all[:,iday,mask0]
        Z2 = np.zeros([2,2])
        γ = np.zeros(2)
        v2 = 0.
        count = 0
        for icmp in range(nb_cmp):
            Σ = np.zeros([3,(te-ts)])
            scale = yt[icmp].dot(ccf_r[icmp][mask0])/ms_r[icmp]
            Z0, Z1  = np.zeros((te-ts)), np.zeros((te-ts))
            A, α = at[0], at[1]#/at[0]
            if scale > 0.4 and scale < 2.:#skip the bad data
                for i in range(3):
                    Σ[i] = ((α+β[itime])*lag_time)**i
                for i in range(3):
                    Z0 += (Tayler_Series[icmp][i]/math.factorial(i)*Σ[i])
                        
                Z1 = A*(Tayler_Series[icmp][1])
                for i in range(1,3):
                    Z1 += A*(Tayler_Series[icmp][i+1]/math.factorial(i)*Σ[i])
                    Z1 += A*(Tayler_Series[icmp][i]/math.factorial(i-1)*Σ[i-1])
                        
                Z1 *= lag_time
                Zt = np.array([Z0.T,Z1.T]).T
                Z2 += Zt.T @ Zt #Z: zeta
                vt = yt[icmp] - A*Z0 #Zt @ at[itime][cmp1][cmp2] #Reduction of β
                γ += Zt.T @ vt
                v2 += vt.T @ vt
                mask_param[itime][icmp] = True
                count += 1            
                    
        Ξ = Pt/h0 - (Pt @ Z2 @ np.linalg.pinv(Z2/h0 +np.linalg.pinv(Pt)) /(h0**2))#Z2: St
        #Likelihood
        λ = np.linalg.eig(Z2 @ Pt)[0]
        lnL1 = np.log(λ[0]+h0)+np.log(λ[1]+h0)+((te-ts)*nb_cmp-2)*np.log(h0)
        lnL2 = (h0*v2-γ.T @ (np.linalg.pinv(Z2/h0 +np.linalg.pinv(Pt))) @ γ)/(h0**2)
        lnL += -(lnL1+lnL2)/2
        att[itime] = at + Ξ @ γ # at + Kt @ vt
        Ptt[itime] = Pt - Ξ @ (Z2 @ Pt @ Z2 + h0 * Z2) @ Ξ.T #Pt - Kt@Ft@Kt.T 
        at[:] = att[itime]      #Tt=1
        Pt[:] = Ptt[itime] + Qt    #Tt and Rt = 1 
        itime += 1
    lnL -= (te-ts)*count/2.*np.log(2*np.pi)
        
    #####################
    ## Kalman Smoother ##
    #####################
    itmax = itime-1
    αt[itmax] = att[itmax]
    Vt[itmax] = Ptt[itmax]
    for itime in reversed(range(itmax)):
        Pt1        = Ptt[itime]+Qt #P_t+1|t
        At         = Ptt[itime]*np.linalg.pinv(Pt1)
        αt[itime] = att[itime]+ At @ (αt[itime+1]-att[itime+1])
        Vt[itime]  = Ptt[itime]+ At @ (Vt[itime+1]-Pt1) @ At.T # P_t|n covariance matrix of the state variavle
                
    return αt, Vt, lnL, mask_param
    

def deri(data, n, delta):
    if n > 0:
        dω = 1./(len(data)*delta)*np.pi*2.
        spctrm = np.fft.rfft(data)
        for i in range(len(spctrm)): spctrm[i] = spctrm[i]*(1j*(i)*dω)**n 
        return np.fft.irfft(spctrm)
    elif n == 0: return(data)
    
def create_lock_file(filename) :
    ff=open(filename,'wb')
    pickle.dump(filename,ff)
    ff.close()

def add_metadata(fout,group_name,inu) : 
    ''' add metadata to the output (=processed) hdf5 file which describe the processing applied 
    it contains the name of the function and the input parameters ''' 
    if group_name in fout:
        dd.dispc('WARNING : ' + group_name + ' already esists : no update in /_metadata','r','b')
    else:
        group=fout.create_group(group_name)
        for ikey in inu : 
            try : 
                dset= np.asarray(inu[ikey])
                group.create_dataset(ikey,data=dset)
            except : 
                group.create_dataset(ikey,data=dset.astype('bytes'))  
    return fout 

def mkdir(out_dir) : 
    if os.path.isdir(out_dir) == False :
        os.makedirs(out_dir)

def load_pkl(filename) : 
    dd.dispc('  loading '+filename,'y','b')
    ff=open(filename,'rb') 
    db=pickle.load(ff)
    ff.close() 
    return db 

