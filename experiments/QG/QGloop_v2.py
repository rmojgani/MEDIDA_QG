import numpy as np 
from random import random

from QGlib_v2 import QGfun
#from QGlib import QGinitialize, QGinitialize2
from QGlib import QGsave
import matplotlib.pyplot as plt
from tqdm import tqdm
from QGlib import QGinitialize3

def QGloop(ts, kk, ll, 
          D_Dx, D_Dy, Grad, Lapl, Grad8,
          psic_1, vorc_1, 
          psic_2, vorc_2, 
          psi_1, psi_2, psi_R, CONSTANTS,
          sponge, q_1, q_2, 
          topographicPV,
          qc_1, qc_2,
          dt, y, g,
          opt,lim,st,
          pv, psiAll, Uall, Vall,
          c_me_jac):#, c_non_linear):
    print('--------> c_me_jac:', c_me_jac )#,', c_non_linear:', c_non_linear)
    print('-------->', CONSTANTS)
    nu = CONSTANTS['nu']#viscous dissipation
    tau_d1 = CONSTANTS['tau_d1']#Newtonian relaxation time-scale for interface
    tau_d2 = CONSTANTS['tau_d2']#Newtonian relaxation time-scale for interface
    tau_f = CONSTANTS['tau_f']#surface friction 
    beta = CONSTANTS['beta']#beta 
    
    len_ = (ts+1 - lim) // st + 1
   
    vorc_1_all, vorc_2_all, q_1_all, q_2_all, qc_1_all, qc_2_all,\
    psic_1_all, psic_2_all, psi_1_all, psi_2_all =\
    QGinitialize3(192, 96, 49, len_)
            
    i_save = 0
    
#    for i in range( 0, ts+1 ):
    for i in tqdm(range(0, ts+1 )):

#        print('Reference step')
#        if i % 1 == 0:
#            print("Timestep:", i, "/", ts)
        
            
#        plt.figure(figsize=(10,2.5))
#        plt.subplot(1,3,1)
#        plt.pcolor(q_1[0,:,:]);plt.colorbar()
#        plt.subplot(1,3,2)
#        plt.pcolor(psi_1[0,:,:]);plt.colorbar()
#        plt.show()
    
        psic_1, psic_2,\
        vorc_1, vorc_2,\
        psi_1 , psi_2 ,\
        q_1   , q_2   ,\
        qc_1  , qc_2  ,\
        x \
          = QGfun(i, kk, ll, D_Dx, D_Dy, Grad, Lapl, Grad8,
                  psic_1, vorc_1, beta, 
                  psic_2, vorc_2, 
                  psi_1, psi_2, psi_R, tau_d1, tau_d2, tau_f,
                  sponge, q_1, q_2, 
                  topographicPV,
                  qc_1, qc_2,
                  dt, nu, y, g,
                  opt,
                  c_me_jac)
          
   

        pv, Uall, Vall, psiAll = 0,0,0,0#QGsave(qc_1, qc_2, i, lim, st, ll, kk, \
                                        #psic_1, psic_2, pv, psiAll, Uall, Vall, psi_1, psi_2)
                                        
#        print('++++++++++++time step:', i)
        if i >= lim and i % st == 0:
#            i_save = (i - lim) // st  - 1
            print(i, i_save,'/',len_)
#
#            plt.figure(figsize=(10,2.5))
#            plt.subplot(1,3,1)
#            plt.pcolor(q_1[0]);plt.colorbar()
#            plt.subplot(1,3,2)
#            plt.pcolor(psi_1[0]);plt.colorbar()
#            plt.subplot(1,3,3)
#            plt.xlabel('in loop')
#            plt.show()
            
            vorc_1_all[:,:,:,i_save] = vorc_1
            vorc_2_all[:,:,:,i_save] = vorc_2
            q_1_all[:,:,:,i_save]  = q_1
            q_2_all[:,:,:,i_save] = q_2
            qc_1_all[:,:,:,i_save] = qc_1
            qc_2_all[:,:,:,i_save] = qc_2
            psic_1_all[:,:,:,i_save] = psic_1
            psic_2_all[:,:,:,i_save] = psic_2
            psi_1_all[:,:,:,i_save]  = psi_1
            psi_2_all[:,:,:,i_save] = psi_2
        
            i_save += 1

#    plt.figure(figsize=(10,2.5))
#    plt.subplot(1,3,1)
#    plt.pcolor(q_1_all[0,:,:,-1]);plt.colorbar()
#    plt.subplot(1,3,2)
#    plt.pcolor(psi_1_all[0,:,:,-1]);plt.colorbar()
#    plt.subplot(1,3,3)
#    plt.xlabel('out loop')
#    plt.show()
        
    return pv, Uall, Vall, psiAll,\
                vorc_1_all , vorc_2_all  ,\
                q_1_all   , q_2_all      ,\
                qc_1_all  , qc_2_all     ,\
                psic_1_all, psic_2_all   ,\
                psi_1_all , psi_2_all
#    return pv, Uall, Vall, psiAll,\
#                vorc_1 , vorc_2  ,\
#                q_1   , q_2      ,\
#                qc_1  , qc_2     ,\
#                psic_1, psic_2   ,\
#                psi_1 , psi_2