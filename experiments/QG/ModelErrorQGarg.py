#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on commits:
    9eb101d
    bdbc2e7
"""

def ModelErrorQGarg():
    import argparse
    parser = argparse.ArgumentParser(description='Model Error QG - Ensemble Kalman Filter')
    
    # Case parameters
    parser.add_argument('--CASE_NO', type=int, default=2, help='Case number - numeric code')
    parser.add_argument('--CASE_NO_R', type=str, default='A', help='Case number - character code')
    
    # Data parameters
    parser.add_argument('--SENSOR_RATIO', type=float, default=0.25, help='Sensor to State ratio')
    parser.add_argument('--NOISE_MAG', type=float, default=0.001, help='Noise magnitude (ratio to ??? )')
    parser.add_argument('--LEN_LOOP', type=int, default=1, help='Length of loop')

    # EnKF parameters
    # parser.add_argument('--N_ENS', type=int, default=2, help='Size of the ensembles')
    # parser.add_argument('--SIG_B', type=float, default=0.002, help='Perturbation')
    
    # Neural Network Parameters, Architecture
    parser.add_argument('--DEEP', type=int, default=4, help='How deep the NN?')
    parser.add_argument('--HIDDEN', type=int, default=1000, help='Case number - numeric code')
    parser.add_argument('--LAMBDA_REG', type=float, default=0.0, help='FFT loss, regularizer weight')
    parser.add_argument('--LOSS_FFT_CUT', type=int, default=0, help='FFT loss, cut-off wave')

    # Neural Network Parameters, Training
    parser.add_argument('--NUM_EPOCHS_SGD', type=int, default=int(1e3*100*2*0), help='SGD iterations')
    parser.add_argument('--NUM_EPOCHS_ADAM', type=int, default=int(1e3*100*2), help='ADAM iterations')
    parser.add_argument('--NUM_EPOCHS_BFGS', type=int, default=int(1e3*0), help='BFGS iterations')
    parser.add_argument('--GAMMA_DIFF', type=float, default=1.0, help='Hyper-parameter for delta-w')
    parser.add_argument('--GAMMA_LAPL', type=float, default=1.0, help='Hyper-parameter for delta-w')
    #parser.add_argument('--LR0', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--LR0', type=float, default=0.01, help='Learning rate')

    # Fourier Feature
    parser.add_argument('--RFF_SIGMA', type=float, default=10.0, help='Random Fourier Feature, sigma')

    return parser.parse_args()
