#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 09:55:49 2021

@author: rmojgani
"""
import numpy as np 
import random
random.seed(0)
#%%
# paper case 
# 1: CASE NO = 3A
# 2: CASE NO = 4A

#%%
def case_select( CASE_NO, CASE_NO_R, xx, yy):
    
    if CASE_NO == 1:
        IF_tau_d_m_ratio = 0
        IF_tau_f_m_ratio = 0
    elif CASE_NO == 2:
        IF_tau_d_m_ratio = 0
        IF_tau_f_m_ratio = 1
    elif CASE_NO == 3:
        IF_tau_d_m_ratio = 1
        IF_tau_f_m_ratio = 0
    elif CASE_NO == 4:
        IF_tau_d_m_ratio = 1
        IF_tau_f_m_ratio = 1



    if CASE_NO_R == 'A':
        H_r = [0.0, 0.0, 0.0, 0.0]
        H_m = [0.0, 0.0, 0.0, 0.0]
    elif CASE_NO_R == 'B':
        H_r = [20, 20, 20, 20.0]
        H_m = [20, 20, 20, 20.0]
    elif CASE_NO_R == 'C':
        H_r = [20.0, 20.0, 20.0, 20.0]
        H_m = [20.0, 10.0, 15.0,  5.0]

    CONSTANTS_r = {
        'nu':pow( 10., -6. ), #viscous dissipation
        'tau_d':0.01,         #Newtonian relaxation time-scale for interface
        'tau_f':0.07,         #surface friction 
        'beta' :0.196,        #beta 
        'sigma':3.5
        }
    
    beta = CONSTANTS_r['beta']
    sigma = CONSTANTS_r['sigma']
    
    CONSTANTS_m = CONSTANTS_r.copy()
    
    CONSTANTS_m['tau_d'] = CONSTANTS_m['tau_d'] + 0.1*IF_tau_d_m_ratio
    CONSTANTS_m['tau_f'] = CONSTANTS_m['tau_f'] + 0.3*IF_tau_f_m_ratio
    
    sx, sy = 5, 5
    
    x_mountain_r = [0, -10,   0,   0]
    y_mountain_r = [0, -5 , +20, -20]
    
    x_mountain_m = x_mountain_r.copy()
    y_mountain_m = y_mountain_r.copy()
    
    topographicPV_ref = my_topography(xx,x_mountain_r,sx,
                                      yy,y_mountain_r,sy,
                                      H_r)
        
    topographicPV_model = my_topography(xx,x_mountain_m,sx,
                                        yy,y_mountain_m,sy,
                                        H_m)
        

    return CONSTANTS_r, topographicPV_ref  , x_mountain_r, y_mountain_r, H_r,\
           CONSTANTS_m, topographicPV_model, x_mountain_m, y_mountain_m, H_m,\
           sigma, beta, sx, sy
#%%
def operator_gen(N, Lx, N2, Ly):
    # Wavenumbers:
    kk = np.fft.rfftfreq( N, Lx / float(N) / 2. / np.pi ) #zonal wavenumbers
    ll = np.fft.fftfreq( N2, Ly / float(N2) / 2. / np.pi ) #meridional wavenumbers
    
    Grad = 1j * ( kk[np.newaxis, :]  + ll[:, np.newaxis] )
    Lapl = -(np.expand_dims(ll, 1) ** 2 + np.expand_dims(kk, 0) ** 2)
    Grad8 = np.expand_dims(kk, 0) ** 8 + np.expand_dims(ll, 1) ** 8
    D_Dx = 1.j * kk[np.newaxis, :]
    D_Dy = 1.j * ll[:, np.newaxis]
    
    return kk, ll, Grad, Lapl, Grad8, D_Dx, D_Dy
#%%
#def QGinitialize(N, N2, ll, kk, beta, y):
def QGinitialize(N, N2, ll, kk, Lapl, beta, y, R):
    
    #  Declare arrays
    
    #Spectral arrays, only need 3 time-steps
    
    psic_1 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )
    psic_2 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )
    qc_1 = np.zeros( ( ( 3 , N2, int(N / 2 + 1) ) ) ).astype( complex )
    qc_2 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )
    vorc_1 = np.zeros( ( ( 3 , N2, int(N / 2 + 1)  ) ) ).astype( complex )
    vorc_2 = np.zeros( ( ( 3 , N2 , int(N / 2 + 1) ) ) ).astype( complex )
    
    print('complex spectral array shapes: ' + str(psic_1.shape))
    
    #Real arrays, only need 3 time-steps
    psi_1 = np.zeros( ( ( 3 , N2 , N ) ) )
    psi_2 = np.zeros( ( ( 3 , N2 , N ) ) )
    q_1 = np.zeros( ( ( 3 , N2, N ) ) )
    q_2 = np.zeros( ( ( 3 , N2 , N ) ) )
    
    print('real array shapes: ' + str(psi_1.shape))

    #  Initial conditions:
    
    psic_1[0] = [ [ random.random() for i in range(int(N / 2 + 1) ) ] for j in range(N2) ]
    psic_2[0] = [ [ random.random() for i in range(int(N / 2 + 1) ) ] for j in range(N2) ]
    
    #Transfer values:
    psic_1[ 1 , : , : ] = psic_1[ 0 , : , : ]
    psic_2[ 1 , : , : ] = psic_2[ 0 , : , : ]
    
    #Calculate initial PV
    for i in range( 2 ):
    	vorc_1[i], vorc_2[i] = ptq(psic_1[i], psic_2[i], Lapl) 
    	q_1[i] = np.fft.irfft2( vorc_1[i]) + beta * y[:, np.newaxis]
    	q_2[i] = np.fft.irfft2( vorc_2[i]) + beta * y[:, np.newaxis]+R 
    	qc_1[i] = np.fft.rfft2( q_1[i] )
    	qc_2[i] = np.fft.rfft2( q_2[i] )
    
#    psi1 = np.fft.irfft2( psic_1[1] )
#    psi2 = np.fft.irfft2( psic_2[1] )
    
    return psic_1, psic_2, vorc_1, vorc_2, q_1, q_2, qc_1, qc_2, psi_1, psi_2
#%%
def QGinitialize2(N, N2, ts, lim, st):
    
    #######################################################
    #  Main time-stepping loop
    
    pv = np.zeros( ( ( ( int((ts - lim) / st), 2, N2, N )  ) ) )
    psiAll = np.zeros( ( ( ( int((ts - lim) / st), 2, N2, N )  ) ) )
    Uall = np.zeros( ( ( ( int((ts - lim) / st), 2, N2, N )  ) ) )
    Vall = np.zeros( ( ( ( int((ts - lim) / st), 2, N2, N )  ) ) )
    
    return pv, psiAll, Uall, Vall
#%%
def QGinitialize3(N2, N, Nc, len_):
    
    vorc_1_all  = np.empty((3, N2, Nc,len_), dtype=complex)
    vorc_2_all  = np.empty((3, N2, Nc,len_), dtype=complex)
    
    q_1_all     = np.empty((3, N2, N,len_))
    q_2_all     = np.empty((3, N2, N,len_))
    
    qc_1_all    = np.empty((3, N2, Nc,len_), dtype=complex)
    qc_2_all    = np.empty((3, N2, Nc,len_), dtype=complex)
    
    psic_1_all  = np.zeros((3, N2, Nc,len_), dtype=complex)
    psic_2_all  = np.empty((3, N2, Nc,len_), dtype=complex)
    
    psi_1_all   = np.empty((3, N2, N,len_))
    psi_2_all   = np.empty((3, N2, N,len_))
    
    return vorc_1_all, vorc_2_all, q_1_all, q_2_all, qc_1_all, qc_2_all,\
            psic_1_all, psic_2_all, psi_1_all, psi_2_all        
#%%
def initialize_psi_to_param(psi_1, psi_2, N2, N, beta, Lapl, y, R):
    #######################################################
    #  Declare arrays
    #######################################################
    # Spectral arrays, only need 3 time-steps
    psic_1 = np.zeros( ( ( N2 , int(N / 2 + 1) ) ) ).astype( complex )
    psic_2 = np.zeros( ( ( N2 , int(N / 2 + 1) ) ) ).astype( complex )
    qc_1   = np.zeros( ( ( N2 , int(N / 2 + 1) ) ) ).astype( complex )
    qc_2   = np.zeros( ( ( N2 , int(N / 2 + 1) ) ) ).astype( complex )
    vorc_1 = np.zeros( ( ( N2 , int(N / 2 + 1) ) ) ).astype( complex )
    vorc_2 = np.zeros( ( ( N2 , int(N / 2 + 1) ) ) ).astype( complex )

    #print('complex spectral array shapes: ' + str(psic_1.shape))

    #Real arrays, only need 3 time-steps
    q_1 = np.zeros( ( N2, N ) )
    q_2 = np.zeros( ( N2, N) )
    #######################################################
    #  
    #######################################################
    psic_1 = np.fft.rfft2(psi_1)
    psic_2 = np.fft.rfft2(psi_2)

    #Calculate initial PV
    vorc_1, vorc_2 = ptq(psic_1, psic_2, Lapl) 
    q_1  = np.fft.irfft2( vorc_1) + beta * y[:, np.newaxis]
    q_2  = np.fft.irfft2( vorc_2) + beta * y[:, np.newaxis] + R 
    qc_1 = np.fft.rfft2( q_1 )
    qc_2 = np.fft.rfft2( q_2 )
       
    return psic_1, psic_2, vorc_1, vorc_2, q_1, q_2, qc_1, qc_2    
#%%
def initialize_psi_to_param_loop(psi_1_M, psi_2_M, N2, N, beta, Lapl, y, topo):

    psi_1_kk, psi_2_kk = {}, {}
    psic_1_kk, psic_2_kk = {}, {}
    vorc_1_kk, vorc_2_kk = {}, {}
    q_1_kk, q_2_kk = {}, {}
    qc_1_kk, qc_2_kk = {}, {}
    
    for kk1 in range(3):
        psi_1_kk[str(kk1)] , psi_2_kk[str(kk1)]  = psi_1_M[kk1] , psi_2_M[kk1]
        
        psic_1_kk[str(kk1)], psic_2_kk[str(kk1)],\
        vorc_1_kk[str(kk1)], vorc_2_kk[str(kk1)],\
        q_1_kk[str(kk1)]   , q_2_kk[str(kk1)],\
        qc_1_kk[str(kk1)]  , qc_2_kk[str(kk1)] =\
        initialize_psi_to_param(psi_1_kk[str(kk1)], psi_2_kk[str(kk1)],\
                                N2, N, beta, Lapl, y, topo)
        
    psi_1 = np.array((psi_1_kk['0'],psi_1_kk['1'],psi_1_kk['2']))
    psi_2 = np.array((psi_2_kk['0'],psi_2_kk['1'],psi_2_kk['2']))
    
    psic_1 = np.array((psic_1_kk['0'],psic_1_kk['1'],psic_1_kk['2']))
    psic_2 = np.array((psic_2_kk['0'],psic_2_kk['1'],psic_2_kk['2']))
    
    vorc_1 = np.array((vorc_1_kk['0'],vorc_1_kk['1'],vorc_1_kk['2']))
    vorc_2 = np.array((vorc_2_kk['0'],vorc_2_kk['1'],vorc_2_kk['2']))
    
    q_1 = np.array((q_1_kk['0'],q_1_kk['1'],q_1_kk['2']))
    q_2 = np.array((q_2_kk['0'],q_2_kk['1'],q_2_kk['2']))
    qc_1 = np.array((qc_1_kk['0'],qc_1_kk['1'],qc_1_kk['2']))
    qc_2 = np.array((qc_2_kk['0'],qc_2_kk['1'],qc_2_kk['2']))
    
    return psi_1, psi_2, psic_1, psic_2, vorc_1, vorc_2, q_1, q_2, qc_1, qc_2

#%%
import scipy.io as sio
def my_load(file_name_read):
    
    pv = sio.loadmat(file_name_read)['pv']
    q_1 = sio.loadmat(file_name_read)['q_1']
    q_2 = sio.loadmat(file_name_read)['q_2']
    qc_1 = sio.loadmat(file_name_read)['qc_1']
    qc_2 = sio.loadmat(file_name_read)['qc_2']
    vorc_1 = sio.loadmat(file_name_read)['vorc_1']
    vorc_2 = sio.loadmat(file_name_read)['vorc_2']
    psi_1 = sio.loadmat(file_name_read)['psi_1']
    psi_2 = sio.loadmat(file_name_read)['psi_2']
    psic_1 = sio.loadmat(file_name_read)['psic_1']
    psic_2 = sio.loadmat(file_name_read)['psic_2']
    
    return pv, q_1, q_2, qc_1, qc_2, vorc_1, vorc_2,\
            psi_1, psi_2, psic_1, psic_2
#%%        
def QGfun(i, kk, ll, D_Dx, D_Dy, Grad, Lapl, Grad8,
          psic_1, vorc_1, beta, 
          psic_2, vorc_2, 
          psi_1, psi_2, psi_R, tau_d, tau_f, 
          sponge, q_1, q_2, 
          R, 
          qc_1, qc_2,
          dt, nu, y, g,
          opt,
          c_me_jac, c_non_linear):

	#NL terms =
    # J(psi, qc+R) 
#    jac11 = calc_jac(D_Dx, D_Dy, psic_1[1, :, :], vorc_1[1, :, :] , c_me_jac ) 
#    jac22 = calc_jac(D_Dx, D_Dy, psic_2[1, :, :], vorc_2[1, :, :]+np.fft.rfft2(R), c_me_jac )
    jac1 = calc_jac1_psi(D_Dx, D_Dy, Lapl, psic_1[1], psic_2[1], 1, 0                , c_me_jac)
    jac2 = calc_jac2_psi(D_Dx, D_Dy, Lapl, psic_1[1], psic_2[1], 2, np.fft.rfft2(R)  , c_me_jac)

    # calc_jac( kk, ll, psic_2[1, :, :], np.fft.rfft2(R) ) 
#    if c_me_jac == 0.5: 
#    # beta * v
    betav1 = beta * D_Dx * psic_1[1, :, :]
    betav2 = beta * D_Dx * psic_2[1, :, :] 

	#Linear terms
	#Relax interface
    forc1 = (psi_1[1] - psi_2[1] - psi_R[:, np.newaxis]) * tau_d 
    forc2 = -(psi_1[1] - psi_2[1] - psi_R[:, np.newaxis]) * tau_d

	#Sponge
    forc1 -= sponge[:, np.newaxis] * (q_1[1] - np.mean( q_1[1], axis = 1)[:, np.newaxis] )
    forc2 -= sponge[:, np.newaxis] * (q_2[1] - np.mean( q_2[1], axis = 1)[:, np.newaxis] )

    #Convert to spectral space
    cforc1 = np.fft.rfft2( forc1 )
    cforc2 = np.fft.rfft2( forc2 ) 
    
    # friction
    cfforc2 = - Lapl * psic_2[1] * tau_f

    rhs1 = -jac1[:] - betav1[:] + cforc1[:] #+ (1-c_non_linear)*calc_nl(psic_1[1],psic_1[1])
    rhs2 = -jac2[:] - betav2[:] + cforc2[:] + cfforc2[:]
    #mrhs = mnl[:]
	
    if i == 0:# changed from 1 to 0
	#Forward step
#        print(i, 'Forward Step')
        qc_1[2, :] = fs(qc_1[1, :, :], rhs1[:], dt, nu, Grad8)
        qc_2[2, :] = fs(qc_2[1, :, :], rhs2[:], dt, nu, Grad8)
        #mc[2, :] = fs(mc[1, :, :], mrhs[:], dt, nu, kk, ll)
    else:
#        print(i, 'Leapfrog step')
	#Leapfrog step
        qc_1[2, :, :] = lf(qc_1[0, :, :], rhs1[:], dt, nu, Grad8)
        qc_2[2, :, :] = lf(qc_2[0, :, :], rhs2[:], dt, nu, Grad8)
        #mc[2, :] = lf(mc[0, :, :], mrhs[:], dt, nu, kk, ll)


    #Convert to real space
    #psi_1[1] = np.fft.irfft2( psic_1[1] )
    #psi_2[1] = np.fft.irfft2( psic_2[1] )
    #u2 = np.fft.irfft2( -1.j * np.expand_dims(ll, 1) * psic_2[1] )
    #v2 = np.fft.irfft2( 1.j * np.expand_dims(kk, 0) * psic_2[1] )
    if i > 1:# changed from 1 to 0
	#Leapfrog filter
        qc_1[1, :] = filt( qc_1[1, :], qc_1[0, :], qc_1[2, :], g)
        qc_2[1, :] = filt( qc_2[1, :], qc_2[0, :], qc_2[2, :], g)
        #mc[1, :] = filt( mc[1, :], mc[0, :], mc[2, :], g)

    for j in range( 2 ):
        q_1[j] = np.fft.irfft2( qc_1[j + 1] )
        q_2[j] = np.fft.irfft2( qc_2[j + 1] )
        
        #Subtract off beta and invert
        vorc_1[j] = np.fft.rfft2( q_1[j] - beta * y[:, np.newaxis]) # RM: vor is --> q1 - \beta y
        vorc_2[j] = np.fft.rfft2( q_2[j] - beta * y[:, np.newaxis]-R) # RM: vor --> q2 - \beta y - R 
        psic_1[j], psic_2[j] = qtp( vorc_1[j], vorc_2[j], ll, kk)
        psi_1[j] = np.fft.irfft2( psic_1[j] ) # RM: psi from spectral domain to real
        psi_2[j] = np.fft.irfft2( psic_2[j] ) # RM: psi from spectral domain to real

        #Transfer values:
        qc_1[j, :, :] = qc_1[j + 1, :, :]
        qc_2[j, :, :] = qc_2[j + 1, :, :]
        
    return psic_1 , psic_2,\
            vorc_1, vorc_2,\
            psi_1 , psi_2 ,\
            q_1   , q_2   ,\
            qc_1  , qc_2  ,'xx'
          

def QGsave(qc_1, qc_2, i, lim, st, ll, kk, psic_1, psic_2, pv, psiAll, Uall, Vall, psi_1, psi_2):

    if i > lim:
        if i % st == 0:
            pv[(i - lim) // st, 0] = np.fft.irfft2( qc_1[2] )
            pv[(i - lim) // st, 1] = np.fft.irfft2( qc_2[2] )
    
    if i > lim:
        if i % st == 0:
            psiAll[(i - lim) // st, 0] = psi_1[1]
            psiAll[(i - lim) // st, 1] = psi_2[1]
    
    if i > lim:
        if i % st == 0:
            Uall[(i - lim) // st, 0] = np.fft.irfft2( -1.j * np.expand_dims(ll, 1) * psic_1[1] )
            Uall[(i - lim) // st, 1] = np.fft.irfft2( -1.j * np.expand_dims(ll, 1) * psic_2[1] )
    
    if i > lim:
        if i % st == 0:
            Vall[(i - lim) // st, 0] = np.fft.irfft2( 1.j * np.expand_dims(kk, 0) * psic_1[1] )
            Vall[(i - lim) // st, 1] = np.fft.irfft2( 1.j * np.expand_dims(kk, 0) * psic_2[1] )
                 
    return pv, Uall, Vall, psiAll

#######################################################
#  Spectral functions

def ptq(ps1, ps2, Lapl):
    """Calculate PV"""
    q1 = Lapl * ps1 - (ps1 - ps2) # -(k^2 + l^2) * psi_1 -(psi_1-psi_2)
    q2 = Lapl * ps2 + (ps1 - ps2) # -(k^2 + l^2) * psi_2 +(psi_1-psi_2)
    
    return q1, q2

def qtp(q1_s, q2_s, ll, kk):
	"""Invert PV"""
	psi_bt = -(q1_s + q2_s) / (ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2) / 2.  # (psi_1 + psi_2)/2
	psi_bc = -(q1_s - q2_s) / (ll[:, np.newaxis] ** 2 + kk[np.newaxis, :] ** 2 + 2. ) / 2.  # (psi_1 - psi_2)/2
	psi_bt[0, 0] = 0.
	psi1 = psi_bt + psi_bc
	psi2 = psi_bt - psi_bc

	return psi1, psi2
          
#######################################################
#  Time-stepping functions

def calc_nl( psi, qc ):
    """"Calculate non-linear terms, with Orszag 3/2 de-aliasing"""
    N2, N = np.shape( psi )
    ex = int(N *  3 / 2)# - 1
    ex2 = int(N2 * 3 / 2)# - 1
    temp1 = np.zeros( ( ex2, ex ) ).astype( complex )
    temp2 = np.zeros( ( ex2, ex ) ).astype( complex )
    temp4 = np.zeros( ( N2, N ) ).astype( complex )	#Final array

    #Pad values:
    temp1[:N2//2, :N] = psi[:N2//2, :N]
    temp1[ex2-N2//2:, :N] = psi[N2//2:, :N]

    temp2[:N2//2, :N] = qc[:N2//2, :N]
    temp2[ex2-N2//2:, :N] = qc[N2//2:, :N]

    #Fourier transform product, normalize, and filter:
    temp3 = np.fft.rfft2( np.fft.irfft2( temp1 ) * np.fft.irfft2( temp2 ) ) * 9. / 4.
    temp4[:N2//2, :N] = temp3[:N2//2, :N]
    temp4[N2//2:, :N] = temp3[ex2-N2//2:, :N]

    return temp4

def calc_jac(D_Dx, D_Dy, psi, qc, c_me_jac):
    """"Calculate Jacobian"""
    dpsi_dx = D_Dx * psi 
    dpsi_dy = D_Dy * psi 

    dq_dx = D_Dx * qc
    dq_dy = D_Dy * qc 

    return  c_me_jac*calc_nl( dpsi_dx, dq_dy ) - c_me_jac*calc_nl( dpsi_dy, dq_dx )

#def calc_jac_psi(D_Dx, D_Dy, Lapl, psic_1, psic_2, k, Rc, c_me_jac):
#    """"Calculate Jacobian"""
#    if k == 1:
#        psic_k = psic_1
#        vorc_k1 = Lapl * psic_k
#        vorc_k2 = -psic_1
#        vorc_k3 =  psic_2
#        q = c_me_jac*(vorc_k1 + vorc_k2 + vorc_k3)
#        
#    elif k==2:
#        psic_k = psic_2
#        vorc_k1 = Lapl * psic_k
#        vorc_k2 =  psic_1
#        vorc_k3 = -psic_2
#        q = c_me_jac*( vorc_k1 + vorc_k2 + vorc_k3 + Rc)
#
#    nl1 = calc_nl( D_Dx * psic_k, D_Dy * q )
#    nl2 = calc_nl( D_Dy * psic_k, D_Dx * q )
#    
##    nlR = calc_nl( D_Dy * psic_k, D_Dx * q )
#    return (nl1 - nl2)
#           

def calc_jac1_psi(D_Dx, D_Dy, Lapl, psic_1, psic_2, k, Rc, c_me_jac):
    """"Calculate Jacobian"""
    vorc_k1 = Lapl * psic_1
    vorc_k2 = -psic_1
    vorc_k3 =  psic_2
    q = c_me_jac*(vorc_k1 + vorc_k2 + vorc_k3)
    nl1 = calc_nl( D_Dx * psic_1, D_Dy * q )
    nl2 = calc_nl( D_Dy * psic_1, D_Dx * q )
    
    return (nl1 - nl2)
           
def calc_jac2_psi(D_Dx, D_Dy, Lapl, psic_1, psic_2, k, Rc, c_me_jac):
    """"Calculate Jacobian"""
    vorc_k1 = Lapl * psic_2
    vorc_k2 =  psic_1
    vorc_k3 = -psic_2
    q = c_me_jac*(vorc_k1 + vorc_k2 + vorc_k3 + Rc)
    nl1 = calc_nl( D_Dx * psic_2, D_Dy * q )
    nl2 = calc_nl( D_Dy * psic_2, D_Dx * q )
    
    return (nl1 - nl2)
           

def fs(ovar, rhs, dt, nu, Grad8):
    """Forward Step: q^t-1 / ( 1 + 2. dt * nu * (k^4 + l^4 ) ) + RHS"""
    mult = dt / ( 1. + dt * nu * Grad8 )
    return mult * (ovar / dt + rhs)

def lf(oovar, rhs, dt, nu, Grad8):
    """Leap frog timestepping: q^t-2 / ( 1 + 2. * dt * nu * (k^4 + l^4 ) ) + RHS"""
    mult = 2. * dt / ( 1. + 2. * dt * nu * Grad8 )
    
    return mult * (oovar / dt / 2. + rhs)

def filt(var, ovar, nvar, g):
	"""Leapfrog filtering"""
	return var + g * (ovar - 2. * var + nvar )


#######################################################
#  Topology

def my_R(x,x0,sx,
         y,y0,sy,
         z):
    return z*np.exp(-(x-x0)**2/(2*sx**2) - (y-y0)**2/(2*sy*2))

def my_topography(xx,x_mountain,sx,
                  yy,y_mountain,sy,
                  H):
    
    topographicPV = np.zeros(np.shape(xx))
    for icount in range(len(H)):
        topographicPV += my_R(xx,x_mountain[icount],sx,
                              yy,y_mountain[icount],sy, 
                              H[icount])
    return topographicPV

#######################################################
#  Library
def psi_lib(N2, N, Grad, Lapl, Grad8,
            D_Dx  , D_Dy,
            psic_1, psic_2, psi_R,
            vorc_1, vorc_2, 
            qc_1  , qc_2  ,
            psi_1 , psi_2 ,
            y     , beta  ):
#    icut = 0
    idx = 6
    X_labels = []
#    XX_2D = np.zeros((7+8, N2, N))
    XX_2D = np.zeros((7+2, N2, N))
#    XX_2D = np.zeros((2, N2, N))
#    XX_2D = np.zeros((7, N2, N))

    # laplacian psi 1:
    X_labels.append(r'0: $∇^2ψ_1$')
    XX_2D[0] = np.fft.irfft2( Lapl * psic_1 )
    # laplacian psi 2:
    X_labels.append(r'1: $∇^2ψ_2$')
    XX_2D[1] = np.fft.irfft2( Lapl * psic_2 )
    # grad psi 1:
    X_labels.append(r'2: $∇ψ_1$')
    XX_2D[2] = np.fft.irfft2( Grad * psic_1 )
    # grad psi 2:
    X_labels.append(r'3: $∇ψ_2$')
    XX_2D[3] = np.fft.irfft2( Grad * psic_2 )
    # psi 1 :
    X_labels.append(r'4: $ψ_1$')
    XX_2D[4] = np.fft.irfft2(psic_1)#psi_1#
    # psi 2 : 
    X_labels.append(r'5: $ψ_2$')
    XX_2D[5] = np.fft.irfft2(psic_2)# psi_2#
    # psi R :
    X_labels.append(r'6: $ψ_r$')
    XX_2D[6] = (  np.zeros_like(psi_2) + psi_R[:, np.newaxis] ).real

    if XX_2D.shape[0] > 7: 
        # J1 :
        X_labels.append(r'6-1: $J1$')
        XX_2D[7] = np.fft.irfft2(calc_jac1_psi(D_Dx, D_Dy, Lapl, psic_1, psic_2, 1, 0, 1))
        # J2 :
        X_labels.append(r'6-2: $J2$')
        XX_2D[8] = np.fft.irfft2(calc_jac2_psi(D_Dx, D_Dy, Lapl, psic_1, psic_2, 2, 0, 1))    

    if XX_2D.shape[0] > 10: 
        # ======================
        #        layer 1
        # ======================
        # return  calc_nl( dpsi_dx, dq_dy ) - calc_nl( dpsi_dy, dq_dx )
        dpsic_dx = D_Dx * psic_1
        dpsic_dy = D_Dy * psic_1 
        #nl1 :
        idx= idx + 1
        X_labels.append(str(idx)+r'$ ∂_x ψ_1 ⨀ ∂_y ∇^2ψ_1$')
    
    #    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dx, D_Dy * (Lapl * psic_1) )  )
        XX_2D[idx] =  np.fft.irfft2( dpsic_dx) * np.fft.irfft2( D_Dy * (Lapl * psic_1) )
        #nl2 :
        idx= idx + 1
        X_labels.append(str(idx)+r'$ ∂_x ψ_1 ⨀ ∂_y ψ_1$')
    #    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dx, D_Dy * psic_1 )  )
        XX_2D[idx] =  np.fft.irfft2( dpsic_dx ) * np.fft.irfft2( D_Dy * psic_1 )
        #nl2 :
        idx= idx + 1
        X_labels.append(str(idx)+r'$ ∂_x ψ_1 ⨀ ∂_y ψ_2$')
    #    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dx, D_Dy *psic_2 )  )
        XX_2D[idx] =  np.fft.irfft2( dpsic_dx ) * np.fft.irfft2( D_Dy *psic_2  )
        #nl2 :
    #    X_labels.append(r'$nl_1(ψ_1,βy)$')
    #    idx= idx + 1
    #    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dx,  D_Dy * np.fft.rfft2( np.zeros_like(psi_2) + beta*y[:, np.newaxis])       )  )   
    #    XX_2D[idx] = np.fft.irfft2( dpsic_dx ) * np.fft.irfft2( D_Dy *  np.fft.rfft2(  np.zeros_like(psi_2) + beta*y[:, np.newaxis])  )
        
        #nl1 :
        idx= idx + 1
        X_labels.append(str(idx)+r'$∂_y ψ_1 ⨀ ∂_x ∇^2ψ_1$')
    #    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dy,D_Dx * Lapl * psic_1 )  )
        XX_2D[idx] = np.fft.irfft2( dpsic_dy ) * np.fft.irfft2( D_Dx * Lapl * psic_1  )
         
    #    #nl2 :
    #    X_labels.append(str(idx)+r'$(∂_y ψ_1 ⨀  ∂_x ψ_1)$')
    #    idx= idx + 1
    ##    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dy,D_Dx * psic_1 )  )
    #    XX_2D[idx] = np.fft.irfft2( dpsic_dy ) * np.fft.irfft2( D_Dx * psic_1 )
        #nl2 :
        idx= idx + 1
        X_labels.append(str(idx)+r'$∂_y ψ_1 ⨀  ∂_x ψ_2$')
    #    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dy,D_Dx * psic_2 )  )
        XX_2D[idx] =  np.fft.irfft2( dpsic_dy ) * np.fft.irfft2( D_Dx * psic_2 )
    #    #nl2 :
    #    X_labels.append(r'$nl(ψ_2,βy)$')
    ##    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dy, D_Dx *   np.fft.rfft2( np.zeros_like(psi_2) + beta*y[:, np.newaxis])       )  )   
    #    XX_2D[idx] = np.fft.irfft2( dpsic_dy ) * np.fft.irfft2( D_Dx *  np.fft.rfft2(  np.zeros_like(psi_2) + beta*y[:, np.newaxis])  )
    
        # ======================
        #        layer 2
        # ======================
        #     return  calc_nl( dpsi_dx, dq_dy ) - calc_nl( dpsi_dy, dq_dx )
        dpsic_dx = D_Dx * psic_2
        dpsic_dy = D_Dy * psic_2 
        #nl1 :
        idx= idx + 1
        X_labels.append(str(idx)+r'$∂_x ψ_2 ⨀ ∂_y ∇^2ψ_1$')
    #    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dx, D_Dy * Lapl * psic_2 )  )
        XX_2D[idx] =  np.fft.irfft2( dpsic_dx ) * np.fft.irfft2(  D_Dy * Lapl * psic_2 )
        #nl2 :
    #    X_labels.append(str(idx)+r'$(∂_x ψ_2 ⨀  ∂_y ψ_1)$')
    #    idx= idx + 1
    ##    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dx, D_Dy * psic_1 )  )
    #    XX_2D[idx] =  np.fft.irfft2( dpsic_dx ) * np.fft.irfft2(  D_Dy * psic_1  )
        #nl2 :
        idx= idx + 1
        X_labels.append(str(idx)+r'$∂_x ψ_2 ⨀ ∂_y ψ_2$')
        XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dx, D_Dy * psic_2 )  )
    #    XX_2D[idx] = np.fft.irfft2( dpsic_dx ) * np.fft.irfft2( D_Dy * psic_2 )
    #    #nl2 :
    #    X_labels.append(r'$nl_1(ψ_2,βy)$')
    #    idx= idx + 1
    ##    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dx,  D_Dy * np.fft.rfft2( np.zeros_like(psi_2) + beta*y[:, np.newaxis])       )  )   
    #    XX_2D[idx] = np.fft.irfft2( dpsic_dx ) * np.fft.irfft2( D_Dy *  np.fft.rfft2(  np.zeros_like(psi_2) + beta*y[:, np.newaxis])  )
    
        #nl1 :
        idx= idx + 1
        X_labels.append(str(idx)+r'$∂_y ψ_2 ⨀ ∂_x ∇^2ψ_2$')
    #    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dy, D_Dx * Lapl * psic_1 )  )
        XX_2D[idx] = np.fft.irfft2( dpsic_dy ) * np.fft.irfft2( D_Dx * Lapl * psic_2 )
    
        #nl2 :
    #    idx= idx + 1
    #    X_labels.append(str(idx)+r'$∂_y ψ_2 ⨀ ∂_x ψ_1$')
    ##    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dy, D_Dx *  psic_1 )  )
    #    XX_2D[idx] = np.fft.irfft2(dpsic_dy ) * np.fft.irfft2(  D_Dx *  psic_1  )
    
    #    #nl2 :
    #    X_labels.append(r'$(∂_y ψ_2 ⨀ ∂_x ψ_2)$')
    #    idx= idx + 1
    ##    XX_2D[idx] = np.fft.irfft2( calc_nl( dpsic_dy, D_Dx * psic_2 )  )
    #    XX_2D[idx] = np.fft.irfft2( dpsic_dy ) * np.fft.irfft2(  D_Dx * psic_2  )
    
        #nl2 :
    #    X_labels.append(r'$nl_2(ψ_2,βy)$')
    ##    XX_2D[22] = np.fft.irfft2( calc_nl( dpsic_dy,  D_Dx *   np.fft.rfft2( np.zeros_like(psi_2) + beta*y[:, np.newaxis])       )  )   
    #    XX_2D[22-icut] = np.fft.irfft2( dpsic_dy ) * np.fft.irfft2( D_Dx *  np.fft.rfft2(  np.zeros_like(psi_2) + beta*y[:, np.newaxis])  )
    
    
    #    #nl2 :
    #    X_labels.append(r'$(ψ_2,ψ_2)$')
    ##    XX_2D[21] = np.fft.irfft2( calc_nl( dpsic_dy, D_Dx * psic_2 )  )
    #    XX_2D[23-icut] = np.fft.irfft2( psic_2 * psic_2  )
        
    return X_labels, XX_2D
#%%
def record_anim(file_name_anim, q_1_M, psi_1_M, q_2_M, psi_2_M, SKIP=5, FPS = 5):
    import matplotlib.animation as manimation
    import matplotlib.pylab as plt

    # Define the meta data for the movie
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', 
                    artist='Matplotlib',
                    comment=file_name_anim)
    writer = FFMpegWriter(fps=FPS, metadata=metadata)
    
    # Initialize the movie
    fig = plt.figure(figsize=(10,12))
    
    # Update the frames for the movie
    with writer.saving(fig, file_name_anim, 200):
        for i in range(0,psi_1_M.shape[-1], SKIP ):

            plt.subplot(2,2,1)
            plt.pcolor(q_1_M[0,:,:,i],vmin=-8,vmax=8);#plt.colorbar()
            plt.subplot(2,2,2)
            plt.pcolor(psi_1_M[0,:,:,i]);#plt.colorbar()
            
            plt.subplot(2,2,3)
            plt.pcolor(q_2_M[0,:,:,i],vmin=-8,vmax=8);#plt.colorbar()
            plt.subplot(2,2,4)
            plt.pcolor(psi_2_M[0,:,:,i]);#plt.colorbar()
            
            writer.grab_frame()