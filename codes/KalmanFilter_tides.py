import numpy as np
import math 
from scipy import interpolate

# The class of the extended Kalman filter
class Klf:
    def init():#Initialize precipitation data
        ## [p0,p1,Am,φm,γ1]
        Klf.q_ref  = np.array([1E-7, 1E-12, 1E-4, 1.,  1E-3])

    def est_h0(ccfs1, ccf_r, nb_cmp):
        h0 = 0.
        mask0 = [True]*(ccfs1.shape[2])
        num   = 0
        for icmp in range(nb_cmp):
            ccfs        = ccfs1[icmp]
            mccfs       = np.ma.masked_array(ccfs,np.isnan(ccfs))
            ccf_r[icmp] = mccfs.mean(axis=0)
            scale       = ccfs[:,mask0].dot(ccf_r[icmp][mask0])/ccf_r[icmp][mask0].dot(ccf_r[icmp][mask0])
            mask        = (scale>0.5) & (scale<5.)
            if math.isnan(np.mean(((mccfs[mask])[:,mask0]- ccf_r[icmp][mask0])**2)):continue
            num+=1
            h0         += np.mean(((mccfs[mask])[:,mask0]- ccf_r[icmp][mask0])**2)
            mccfs[mask] = np.diag(1./scale[mask]) @ np.array(mccfs[mask])
            ccf_r[icmp] = mccfs[mask].mean(axis=0)

        h0 /= num
        return(h0, ccf_r)
    

    def est_h0_2(ccfs1, ccf_r, nb_cmp,ts,te):
        h0 = 0.
        mask0 = [False]*(ccfs1.shape[2])
        for i in range(ts,te): mask0[i] = True
        num = 0
        for icmp in range(nb_cmp):
            ccfs        = ccfs1[icmp]
            mccfs       = np.ma.masked_array(ccfs,np.isnan(ccfs))
            scale       = ccfs[:,mask0].dot(ccf_r[icmp][mask0])/ccf_r[icmp][mask0].dot(ccf_r[icmp][mask0])
            mask        = (scale>0.5) & (scale<2.)
            if math.isnan(np.mean(((mccfs[mask])[:,mask0]- ccf_r[icmp][mask0])**2)):continue
            num+=1
            h0         += np.mean(((mccfs[mask])[:,mask0]- ccf_r[icmp][mask0])**2)
            mccfs       = np.diag(1./scale[mask]) @ np.array(mccfs[mask])

        h0 /= num    
        return h0
    
    def cal_ref(ccfs_all,alpha_t,delta):
        at = alpha_t[:,1]
        A  = alpha_t[:,0]
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
            Tayler_Series[icmp] = [Klf.deri(ccf_r[icmp], i, delta)[mask0] for i in range(5)]
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
                A, α = at[0], at[1]              

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
                        
            Ξ = Pt/h0 - (Pt @ Z2 @ np.linalg.inv(Z2/h0 +np.linalg.inv(Pt)) /(h0**2))#Z2: St
            #Likelihood
            λ = np.linalg.eig(Z2 @ Pt)[0]
            lnL1 = np.log(λ[0]+h0)+np.log(λ[1]+h0)+((te-ts)*nb_cmp-2)*np.log(h0)
            lnL2 = (h0*v2-γ.T @ (np.linalg.inv(Z2/h0 +np.linalg.inv(Pt))) @ γ)/(h0**2)
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
            αt[itime] = att[itime]+ At @ (αt[itime+1]-att[itime])
            Vt[itime]  = Ptt[itime]+ At @ (Vt[itime+1]-Pt1) @ At.T # P_t|n covariance matrix of the state variavle
                    
        return αt, Vt, lnL, mask_param


    def deri(data,n,delta):
        if n > 0:
            dω = 1./(len(data)*delta)*np.pi*2.
            spctrm = np.fft.rfft(data)
            for i in range(len(spctrm)): spctrm[i] = spctrm[i]*(1j*(i)*dω)**n 
            return np.fft.irfft(spctrm)
        elif n == 0: return(data)
        
        
