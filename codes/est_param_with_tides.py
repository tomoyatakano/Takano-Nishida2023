################################################
# est_param_with_tides.py
# Tomoya Takano (Hirosaki Univ.)
# March 13, 2024
################################################

import time
import pickle
import h5py as h5 
import numpy as np
import scipy.io as io
from obspy import UTCDateTime as UTC

import scipy.signal as signal

from obspy.geodetics import gps2dist_azimuth as gps2dist
from obspy.geodetics import kilometer2degrees
from obspy.geodetics import degrees2kilometers
from obspy.geodetics import locations2degrees
from KalmanFilter_tides import Klf
from scipy import optimize

from obspy.signal.detrend import polynomial
from obspy.core.inventory.inventory import read_inventory
from obspy import read

D2R             = np.pi/180.
tides_deg       = 1.9322736*14.999999

def likelihood(qm,flag=0, ref=0):
    q0              = qm * Klf.q_ref    
    St = q0[2] * np.cos((date_vec[:] * tides_deg + tide_ph + q0[3])*D2R)

    alpha_t, Vt, lnL, mask_param = Klf.eKlf(nb_compo,ccfs, ccf_ref2, tau, ts, te, St, h0, [q0[0], q0[1]], Pt=np.array([[q0[0],0], [0,q0[1]]]), at0 = [1, q0[4]])
    
    if flag == 0:
        np.set_printoptions(precision=2,linewidth=200)
        np.set_printoptions(precision=8)
        return(-lnL+ref)
        
    if flag == 1:
        return (alpha_t, Vt, lnL, St, mask_param)

Klf.init()

with open("testdata_SIKH.pickle",mode="rb") as fi: data_all = pickle.load(fi)
data     = data_all["ccfs"]
lagtime  = data_all["time"]
tau      = data_all["tau"]
date     = data_all["date1"]
nb_date  = len(date)
date_vec = np.arange(1,nb_date)
nb_compo = 9
m0       = int(np.floor(len(lagtime[:])*0.5))
fe       = 1./tau
ts       = int(2/tau)
te       = int(15/tau)
win      = np.zeros(len(time))  
win2     = signal.tukey(te-ts, 0.1)
cc_maxlag = 256
ccfs     = np.zeros((nb_compo, nb_date, len(time)), dtype="float32")
beta     = np.zeros(len(ccfs[0])+1)

for i in range(ts, te): win[i] = win2[i-ts]

for icmp in range(nb_compo):
    for idates in range(nb_date):
        khour  = UTC(date[idates]).hour
        if khour == 9:
            trace = np.zeros((len(time)))
        else:
            trace = data[idates,m0:m0+int(cc_maxlag*fe),icmp]
        ccfs[icmp, idates, :] = trace*win

h0, ccf_ref = Klf.est_h0(ccfs, nb_compo,ts, te, nb_date)
alpha_t, Vt, lnL, mask_param = Klf.eKlf(nb_compo, ccfs, ccf_ref, tau, ts, te, beta, h0,  [Klf.q_ref[0], 1E-10])

ccf_ref2 =  Klf.cal_ref(ccfs,alpha_t,tau)
h0 = Klf.est_h0_2(ccfs,ccf_ref2, nb_compo, ts,te,nb_date)
q_init   = [1.,1.,1.,1.,1.]
bounds   = [[1E-3, 1E3], [0.2,100], [1E-1, 1E2], [-180,180], [-10,10]]

q_est, min, log = optimize.fmin_l_bfgs_b(lambda q_tmp: likelihood([q_tmp[0], q_tmp[1], q_tmp[2], q_tmp[3],q_tmp[4]],Klf.q_ref, date_vec,nb_compo,ccfs, ccf_ref, h0, tau, ts, te, tide_amp,tide_ph, ref=lnL), q_init, epsilon=1e-5,bounds=bounds, approx_grad = True)

alpha_t, Vt, lnL, model, mask_param = likelihood(q_est, Klf.q_ref,date_vec,nb_compo,ccfs, ccf_ref, h0, tau, ts, te, tide_amp, tide_ph, flag=1)

dvv  = alpha_t[:,1]*1E2
err  = np.sqrt(Vt[:,1,1])*1E2

data_all = {"dvv":dvv, "err": err, "beta":q_est*Klf.q_ref, "h0":h0, "model":model}
with open("snet_seaheight.pickle",mode="wb")  as fo:
    pickle.dump(data_all,fo)