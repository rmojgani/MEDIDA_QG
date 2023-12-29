import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

#https://newbedev.com/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set
# def myplot(x, y, s, bins=1000):
#     heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
#     heatmap = gaussian_filter(heatmap, sigma=s)

#     extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
#     return heatmap.T, extent


# fig, axs = plt.subplots(2, 2)

# # Generate some test data
# #xplot = np.random.randn(1000)
# #yplot = np.random.randn(1000)
# xplot = xxv.flatten()
# yplot = yyv.flatten()
# splot = psi_1.flatten()

# sigmas = [0, 1.6, 3.2, 6.4]

# for ax, s in zip(axs.flatten(), sigmas):
#     if s == 0:
#         ax.plot(xplot, yplot, 'k.', markersize=5)
#         ax.set_title("Scatter plot")
#     else:
#         img, extent = myplot(xplot, yplot, splot)
#         ax.imshow(img, extent=extent, origin='lower', cmap='bwr')
#         ax.set_title("Smoothing with  $\sigma$ = %d" % s)

# plt.show()
#%%
def myplot2(varplot,varplotf,label='', IF_SAVE=False):
    dvarplot = varplot-varplotf

    plt.figure(figsize=(14,8))
    
    plt.subplot(2,3,1)
    plt.pcolor(varplot,vmin=-5,vmax=5);plt.colorbar()
    plt.title(label)
    plt.subplot(2,3,2)
    plt.pcolor(varplotf,vmin=-5,vmax=5);plt.colorbar()
    
    # plt.subplot(2,3,6)
    # plt.pcolor(dvarplot,vmin=-0.001,vmax=0.001);plt.colorbar()
    
    VMIN = dvarplot.min()
    VMAX = dvarplot.max()
    plt.subplot(2,3,3)
    plt.pcolor(dvarplot,vmin=VMIN,vmax=VMAX);plt.colorbar()
    plt.title(str(np.round(np.linalg.norm(dvarplot,2),4)))
    
    if IF_SAVE:
        plt.savefig(label+'.png', bbox_inches='tight' , pad_inches = 0)

    plt.show()
    # plt.figure();
    # plt.semilogy(np.fft.rfft2(varplot).ravel(),'.k', markersize=2);
    # plt.semilogy(np.fft.rfft2(varplotf).ravel(),'+r', markersize=2);
    # plt.ylim(bottom=1e-2)
#%%
def smoother1(var, IF_SMOOTH=True):
    if IF_SMOOTH == True:
        import pylops
        N, M = var.shape
        nsmooth1, nsmooth2 = 5, 5
        
        Sop = pylops.Smoothing2D(nsmooth=[nsmooth1, nsmooth2], dims=[N, M], dtype="float64")
        var = Sop * var.ravel()
        var = np.reshape(var, (N, M))
    
    return var

def smoother(var):
    from scipy import fftpack
    im_fft = fftpack.fft2(var)
    keep_fraction = 0.15
    # keep_fraction = 0.20
    # keep_fraction = 0.25

    
    im_fft2 = im_fft.copy()
    
    # Set r and c to be the number of rows and columns of the array.
    r, c = im_fft2.shape
    
    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
    
    # Similarly with the columns:
    im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    

    return  fftpack.ifft2(im_fft2).real

def smoother1(X):
    from sklearn.utils.extmath import randomized_svd

    U, Sigma, VT = randomized_svd(X, 
                                  n_components=50,
                                  n_iter=50,
                                  random_state=None)

    return U@np.diag(Sigma)@VT

def hypervis(X,kk,ll,nu=1e-4):
    dt = 0.025
    Xc = np.fft.rfft2( X )
    kxx = np.expand_dims(kk, 0) ** 2
    kyy = np.expand_dims(ll, 1) ** 2
    # q_hyper=np.zeros([2,np.size(Q,1),np.size(Q,2)],dtype=complex)
#    Q[0,:,:]=np.multiply(np.squeeze(Q[0,:,:]),np.exp(-dt*(nu*np.sqrt((kxx)))**8-dt*(nu*np.sqrt(kyy))**8))
#    Q[1,:,:]=np.multiply(np.squeeze(Q[1,:,:]),np.exp(-dt*(nu*np.sqrt((kxx)))**8-dt*(nu*np.sqrt(kyy))**8))
    XCf = np.multiply(Xc,np.exp(-nu*dt*  (   np.sqrt(kxx+kyy)  )**8))
    
    return np.fft.irfft2( XCf )
# #%%#
# from scipy.io import loadmat
# # folder_path_load = 'save/EnKFSVD/'
# # file_name_load = '2A_RVM_NOISE_MAG=0.001_SIG_B=0.01_N_ENS=100'
# # folder_path_load = 'save/EnKFSVDKK/'
# # file_name_load = '2A_RVM_NOISE_MAG=0.001_SIG_B=0.01_N_ENS=50000' # smoother off: $∇^2ψ_2$ : 0.30, on: $∇^2ψ_2$ : 0.19 ; qon:$∇^2ψ_2$ : 0.19
# # file_name_load = '4A_RVM_NOISE_MAG=0.01_SIG_B=10.0_N_ENS=50000'
# # file_name_load = '3A_RVM_NOISE_MAG=0.01_SIG_B=1.0_N_ENS=50000'
# # file_name_load = '3A_RVM_NOISE_MAG=0.01_SIG_B=0.1_N_ENS=50000'
# # folder_path_load = 'save/EnKFnoise/'
# # file_name_load = '2A_RVM_NOISE_MAG=0.01_SIG_B=0.1_N_ENS=10000'
# folder_path_load = 'save/EnKFSVDKKhyp/'
# # file_name_load = '4A_RVM_NOISE_MAG=0.01_SIG_B=0.1_N_ENS=50000_hyp100'
# # file_name_load = '4A_RVM_NOISE_MAG=0.01_SIG_B=0.1_N_ENS=50000_hyp1000'
# # file_name_load = '4A_RVM_NOISE_MAG=0.01_SIG_B=10.0_N_ENS=50000_hyp100'
# # file_name_load = '4A_RVM_NOISE_MAG=0.01_SIG_B=10.0_N_ENS=50000_hyp10'
# # file_name_load = '4A_RVM_NOISE_MAG=0.01_SIG_B=0.02_N_ENS=50000_hyp10'
# file_name_load = '4A_RVM_NOISE_MAG=0.01_SIG_B=0.1_N_ENS=50000_hyp1'

# psi_1_imperfect_load = loadmat(folder_path_load+file_name_load,variable_names=['psi_1_imperfect'])['psi_1_imperfect']#[::-1,:]
# psi_2_imperfect_load = loadmat(folder_path_load+file_name_load,variable_names=['psi_2_imperfect'])['psi_2_imperfect']#[::-1,:]
# psi_1_observed_load = loadmat(folder_path_load+file_name_load,variable_names=['psi_1_observed'])['psi_1_observed']#[::-1,:]
# psi_2_observed_load = loadmat(folder_path_load+file_name_load,variable_names=['psi_2_observed'])['psi_2_observed']#[::-1,:]

# # psi_1_imperfect_load = psi_1_imperfect
# # psi_2_imperfect_load = psi_2_imperfect
# # psi_1_observed_load = psi_1_observed
# # psi_2_observed_load = psi_2_observed

# # X_labels = loadmat(folder_path_load+file_name_load,variable_names=['X_labels'])['X_labels']#[::-1,:]
# XXts = loadmat(folder_path_load+file_name_load,variable_names=['XXt'])['XXt']#[::-1,:]
# # ##%%
# # psi_1_observed_load= smoother(psi_1_observed_load)
# # psi_2_observed_load = smoother(psi_2_observed_load)
# # psi_1_imperfect_load = smoother(psi_1_imperfect_load)
# # psi_2_imperfect_load = smoother(psi_2_imperfect_load)

# nu=1e-4#*1e-1
# psi_1_observed_load = hypervis(psi_1_observed_load, kk, ll, nu)
# psi_2_observed_load = hypervis(psi_2_observed_load, kk, ll, nu)
# psi_1_imperfect_load = hypervis(psi_1_imperfect_load, kk, ll, nu)
# psi_2_imperfect_load = hypervis(psi_2_imperfect_load, kk, ll, nu)

# _, _, _, _, q_1_observed_load, q_2_observed_load, _, _ =\
# initialize_psi_to_param(psi_1_observed_load, psi_2_observed_load,\
#                             N2, N, beta, Lapl, y, topographicPV_ref)

# _, _, _, _, q_1_imperfect_load, q_2_imperfect_load, _, _ =\
# initialize_psi_to_param(psi_1_imperfect_load, psi_2_imperfect_load,\
#                             N2, N, beta, Lapl, y, topographicPV_model)
    
# plt.pcolor( q_2_observed_load - q_2_imperfect_load );plt.colorbar()

# # q_1_observed_load = smoother(q_1_observed_load)
# # q_2_observed_load = smoother(q_2_observed_load)
# # q_1_imperfect_load = smoother(q_1_imperfect_load)
# # q_2_imperfect_load = smoother(q_2_imperfect_load)

# myplot2(q_1_imperfect,q_1_imperfect_load,'${q_1}_m$',False)
# myplot2(q_2_imperfect,q_2_imperfect_load,'${q_2}_m$',False)
# myplot2(q_1_observed,q_1_observed_load,'${q_1}_o$',False)
# myplot2(q_2_observed,q_2_observed_load,'${q_2}_o$',False)

# # myplot2(psi_1_imperfect,psi_1_imperfect_load,'${\psi_1}_m$')
# # myplot2(psi_2_imperfect,psi_2_imperfect_load,'${\psi_2}_m$')
# # myplot2(psi_1_observed,psi_1_observed_load,'${\psi_1}_o$')
# # myplot2(psi_2_observed,psi_2_observed_load,'${\psi_2}_o$')

# ##%%
# dq1s = np.array([])
# dq2s = np.array([])
# #--------------------------------------------------------------------------
# # Build cost function
# #------------------- 
# #    dq1M = (1/tot_time) * ( q_1_m_filtered - psi_m[:N*N2].reshape(N2, N) )# to check the 0th index
# dq1M_load = (1/tot_time) * ( q_1_observed_load - q_1_imperfect_load )# to check the 0th index
# dq1ns = cutandvectorize(dq1M_load, BOUNDARY_CUT)
# dq1s = np.hstack((dq1s,dq1ns))

# #    dq2M = (1/tot_time) * ( q_2_m_filtered - psi_m[N*N2:].reshape(N2, N) )# to check the 0th index
# dq2M_load = (1/tot_time) * ( q_2_observed_load - q_2_imperfect_load )# to check the 0th index
# dq2ns = cutandvectorize(dq2M_load, BOUNDARY_CUT)
# dq2s = np.hstack((dq2s,dq2ns))

# LAYER = 2
# if LAYER == 1:
#     dqs = dq1s
# elif LAYER == 2:
#     dqs = dq2s

# TOL = 1e-2
# THRESHOLD_ALPHA = 1e3
# #X_labels = range( XX.shape[1])
# clf = RVR(threshold_alpha= THRESHOLD_ALPHA, tol=TOL, verbose=True, standardise=True)
# # fitted = clf.fit(XXt, dq, X_labels);
# # fitted = clf.fit(XXts, dqs, X_labels);
# fitted = clf.fit(XXtsc, dqs, X_labelsc);

# c_rvm_jac = fitted.m_
# #%%
# # dq1s = loadmat(folder_path_load+file_name_load,variable_names=['dq1'])['dq1']#.flatten()
# # dq2s = loadmat(folder_path_load+file_name_load,variable_names=['dq2'])['dq2'].flatten()
# #%%
# plt.subplot(2,3,1)

# qplot = dq2ns
# plt.pcolor(qplot.reshape(*xx_cut.shape));plt.colorbar()
# ##%%
# mdic = {'qplot': qplot
#         }
# savemat("qsave.mat", mdic)

# dqsavemat = loadmat("qsave.mat",variable_names=['qplot'])['qplot'].flatten()

# plt.subplot(2,3,2)
# plt.pcolor(dqsavemat.reshape(*xx_cut.shape));plt.colorbar()

# plt.subplot(2,3,3)
# plt.pcolor((qplot-dqsavemat).reshape(*xx_cut.shape));plt.colorbar()

# #%%
# plt.pcolor(dqs.reshape(*xx_cut.shape));plt.colorbar()