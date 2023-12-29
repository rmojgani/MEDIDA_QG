#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 12:38:37 2022

@author: rm99
"""
def smoother(var,KEEP_FRACTION):
    from scipy import fftpack
    im_fft = fftpack.fft2(var)

    im_fft2 = im_fft.copy()
    
    # Set r and c to be the number of rows and columns of the array.
    r, c = im_fft2.shape
    
    DROP_FRACTION = 1.0 - KEEP_FRACTION 
    rmid = int(r/2)
    rk = int(rmid*DROP_FRACTION)

    cmid = int(c/2)
    ck = int(cmid*DROP_FRACTION)

    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[rmid-rk:rmid+rk,:] = 0
    
    # Similarly with the columns:
    im_fft2[:,cmid-ck:cmid+ck] = 0
    
    #from matplotlib.colors import LogNorm
    #plt.imshow(np.abs(im_fft2), norm=LogNorm(vmin=5));plt.colorbar()
    
    return  fftpack.ifft2(im_fft2).real
#%%
# ifplot = 'NN'
# # ifplot = 'Clean data'

# if ifplot=='NN':
#     var = psi_10
# else:
#     var = psi_1
    
# KEEP_FRACTION=1.0
# from scipy import fftpack
# im_fft = fftpack.fft2(var)

# im_fft2 = im_fft.copy()

# # Set r and c to be the number of rows and columns of the array.
# r, c = im_fft2.shape

# # Set to zero all rows with indices between r*keep_fraction and
# # r*(1-keep_fraction):
# im_fft2[int(r*KEEP_FRACTION):int(r*(1-KEEP_FRACTION))] = 0

# # Similarly with the columns:
# im_fft2[:, int(c*KEEP_FRACTION):int(c*(1-KEEP_FRACTION))] = 0

# plt.title(ifplot)
# plt.loglog(np.abs(im_fft).reshape(-1,1), marker='.')
# plt.ylim([1e-6,1e4])
# #%%
# myplot2(psi_10,psi_1,label=myiter, IF_SAVE=False)
#%%
import numpy as np
import matplotlib.pylab as plt

def plotfft(var1, var10, title=''):
# try:
    # var1= psi_1
    # var10= psi_10
    
    # imin = 0#75
    # imax = 192#110
    imin = 75
    imax = 110
    a10 = 0*np.abs(np.fft.rfft(var10[imin,:]))
    a1 = 0*np.abs(np.fft.rfft(var1[imin,:]))
    
    for ydirc in range(imin,imax):
    # for ydirc in range(0,125):
    
        # np.fft.rfft(var[ydirc,:])
        a10 = a10 + np.abs(np.fft.rfft(var10[ydirc,:]))
        a1 = a1 + np.abs(np.fft.rfft(var1[ydirc,:]))
    
    plt.figure(dpi=250)
    plt.semilogy(a10/(110-75),':r', label='NN')
    plt.semilogy(a1/(110-75),'-k', label='Exact', linewidth=4, alpha=0.25)
    # plt.title(myiter)
    plt.grid(b=True, which='major', color='k', linestyle='-', alpha=0.25)
    plt.grid(b=True, which='minor', color='r', linestyle=':', alpha=0.10)
    plt.minorticks_on()
    
    plt.ylim([1e-3,1e3])
    plt.legend(loc="upper right")

    plt.xlabel(r'$\kappa$',fontsize=12)
    plt.ylabel(r'$E$',fontsize=12)
    plt.title(title)
    plt.show()
    # plot the same for the derivatives 
# except:
#     print('ff')