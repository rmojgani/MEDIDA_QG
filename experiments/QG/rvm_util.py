#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:17:14 2021

@author: rmojgani
"""
import numpy as np
#from PDE_FIND import build_linear_system
from PDE_FIND_edit import build_linear_system
import matplotlib.pyplot as plt

def rvm_cost_function_noise(u_observed, u_imperfect, N,
                      dt, dx,
                      D, P, 
                      time_diff='FD', space_diff='FD'):#,
                      #if_SP=False):
    
    #-------------------
    # Build library
    #-------------------  
    my_linear_system = lambda u: build_linear_system(u, dt, dx, 
                                                     D, P, 
                                                     time_diff, space_diff)
    _, R_observed_r, rhs_des_observed  = my_linear_system(u_observed)
    yy_here = (1.0/dt)*(u_observed-u_imperfect.reshape(N,1))
    dR = R_observed_r.real
    
    return yy_here, dR, rhs_des_observed 

def rvm_cost_function(u_truth, u_imperfect, N,
                      dt, dx,
                      D, P, 
                      time_diff='FD', space_diff='FD'):#,
                      #if_SP=False):
    
    #-------------------
    # Build library
    #-------------------  
    my_linear_system = lambda u: build_linear_system(u, dt, dx, 
                                                     D, P, 
                                                     time_diff, space_diff)
    #%
    _, _, rhs_des_observed = my_linear_system(u_truth.reshape(N,1))
    yy_here = (1.0/dt)*(u_truth-u_imperfect)
    #%
    _, R_observed_now_r, _ = my_linear_system(u_truth.reshape(N,1))
#    if if_SP:
#        _, R_observed_now_r, _ = build_linear_system_SP(u_truth.reshape(N,), kappa)    
    dR = R_observed_now_r.real
    
    return yy_here, dR, rhs_des_observed

def rvm_stack(N, dR, yy_here, XX, yy):
    #-------------------
    # Stacking
    #------------------- 
    if XX.size == 0:
        XX = dR
        yy = yy_here.reshape(N,1)#
    else:
        XX = np.vstack([XX, dR.real])
        yy = np.vstack([yy, yy_here.reshape(N,1)])
        
    return XX, yy


def rvm_add_noise(u, noise_mag=0, noise_std=0, SEED=0):
    #-------------------
    # Add noise
    #-------------------  
    return u + noise_mag*noise_std*np.random.randn(*u.shape)


def rvm_print_model(fitted,rhs_des,if_plot=True):
    rvm_str = ''
    count_term = 0
    count_retained = 0 
    for term in  list(rhs_des):
    
        if fitted.retained_[count_term] == True:
            coeff = fitted.m_[count_retained]
            if coeff > 0:
                rvm_str = rvm_str + '+'
            rvm_str =  rvm_str + str(np.round(coeff,4))+ np.array2string(term)[1:-1]
            count_retained = count_retained+1
            
        count_term = count_term+1
        
    
    rvm_str_r = '"$'+rvm_str+'$"'

    if if_plot:
        plt.figure(figsize=(6, 0.15))
        ax1 = plt.axes(frameon=False)
        ax1.set_frame_on(False)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        plt.title(rvm_str_r)
        
    return rvm_str, rvm_str_r
    
def rvm_print_model_full(fitted, rhs_des, 
                         KS_str, lambdas_imperfect,
                         if_plot=True):

    rvm_str = ''
    count_term = 0
    count_retained = 0 
    
    for term in  list(rhs_des):
        term_in_true_model = np.where(KS_str==term)[0]
        
        if term_in_true_model.shape[0] == 1:
            coeff_model = lambdas_imperfect[term_in_true_model[0]]
        else:
            coeff_model = 0
            
        if fitted.retained_[count_term] == True:
            coeff = - fitted.m_[count_retained] + coeff_model
            
            if coeff > 0:
                rvm_str = rvm_str + '+'
            rvm_str =  rvm_str + str(np.round(coeff,4))+ np.array2string(term)[1:-1]
            count_retained = count_retained+1
        elif coeff_model != 0:
            if coeff_model > 0:
                rvm_str = rvm_str + '+'
            rvm_str =  rvm_str + str(np.round(coeff_model,4))+ np.array2string(term)[1:-1]
    
            
        count_term = count_term+1
        
    
    rvm_str_r = '"$'+rvm_str+'$"'

    if if_plot:
        plt.figure(figsize=(6, 0.15))
        ax1 = plt.axes(frameon=False)
        ax1.set_frame_on(False)
        ax1.axes.get_yaxis().set_visible(False)
        ax1.axes.get_xaxis().set_visible(False)
        plt.title(rvm_str_r)
        
    return rvm_str, rvm_str_r


#%%
def rvm_cost_function_noiseNN(u_observed, u_imperfect, N,
                      dt, dx,
                      D, P,
                      net, xx, tt, scale, 
                      time_diff='FD', space_diff='FD'):#,
                      #if_SP=False):
    
    #-------------------
    # Build library
    #-------------------  
    my_linear_system = lambda u: build_linear_system(u, dt, dx, 
                                                     D, P, 
                                                     time_diff, space_diff)
#    _, R_observed_r, rhs_des_observed  = my_linear_system(u_observed)
    _, _, rhs_des_observed  = my_linear_system(u_observed)
    
    lib_DP = net.lib_DP(xx,tt,scale)
    R_observed_r = np.zeros([u_observed.shape[0],len(lib_DP)])

    for column in range(len(lib_DP)):
        R_observed_r[:,column] = lib_DP[column].detach().cpu().numpy()[:,0]
    
    yy_here = (1.0/dt)*(u_observed-u_imperfect.reshape(N,1))
    dR = R_observed_r.real
    
    return yy_here, dR, rhs_des_observed 