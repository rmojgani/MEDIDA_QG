def init_net(SENSOR_RATIO, N, N2, D_Dx, D_Dy, Lx, Ly):
    import numpy as np
    import torch
    #from mynet2D import Net

    #torch.manual_seed(100)
    #%%
    Nxy = 1.0
    Nx = int(Nxy*N);
    Ny = int(Nxy*N2);
    Nxy = Nx*Ny
    #Niy = Nx*Ny#2016
    a = -np.pi
    b = +np.pi

    #HIDDEN = 1000#int(Nxy)
    #DEEP = 4
    #LR0 = 0.001
    #SENSOR_RATIO = 0.25
    #SENSOR_RATIO = 0.25

    #NUM_EPOCHS_SGD = int(1e3*100*2*0) #233.62it/s (SENSOR_RATIO = 0.75 )
    #NUM_EPOCHS_ADAM = int(1e3*100*2)  # 190.32it/s(SENSOR_RATIO = 0.75 )
    #NUM_EPOCHS_BFGS = int(1e3*0)
    #%%
    X = np.linspace(a,b,Nx,endpoint=True).reshape((Nx,1))
    Y = np.linspace(a,b,Ny,endpoint=True).reshape((Ny,1))

    xv, yv = np.meshgrid(X, Y)
    #%%  in main code : kk, ll, Grad, Lapl, Grad8, D_Dx, D_Dy = operator_gen(N1, Lx, N2, Ly)
    scale = {'x': (b-a)/Lx,
             'y': (b-a)/Ly}
    #%%
    #x = torch.from_numpy(xv.reshape(N,1)).float()
    #x = torch.tensor(x,requires_grad=True)
    #x = x.clone().detach().requires_grad_(True)

    #y = torch.from_numpy(yv.reshape(N,1)).float()
    #y = torch.tensor(y,requires_grad=True)
    #y = y.clone().detach().requires_grad_(True)

    #print(x.shape, y.shape)
    #
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #%% Pick sensor locations
    import random
    random.seed(0)
    #idx = np.sort( random.sample(range(N), int((SENSOR_RATIO)*N)) ).flatten()
    MASK_RATIO = 1.0 - SENSOR_RATIO
    unmask_ratio = 1.0 - MASK_RATIO
    ind_unmask = np.array(random.sample(range(Nxy), int(unmask_ratio*Nxy)))
    ind_unmask = np.sort(ind_unmask, axis=-1)
    #ind_unmask = np.concatenate(([0],ind_unmask,[N-1]),axis=-1) # Add for SENSOR_RATIO!=1
    #%% Masking data
    indices = np.linspace(0, Nxy-1, Nxy, endpoint=True, dtype=int)
    # Masked: True, Unmasked: False
    indices[ind_unmask] = False
    indices[indices!=False] = True
    #%%
    ix = list(np.floor(np.linspace(0, N-1,Nx)).astype('int'));
    iy = list(np.floor(np.linspace(0,N2-1,Ny)).astype('int'));
    #iy1 = list(np.arange(76,124,1).astype('int'));
    #iy.extend(iy1)
    #iy.sort()
    #iy = list(set(iy))# Remove duplicates
    #Niy = 2016
    #%%
    XX = np.linspace(a,b,N,endpoint=True).reshape((N,1))
    YY = np.linspace(a,b,N2,endpoint=True).reshape((N2,1))

    xxv, yyv = np.meshgrid(XX, YY)
    #%%
    xv1 = xxv[iy,:][:,ix].reshape(-1,1)[ind_unmask]
    # x_train = torch.from_numpy(xv1).float()
    # x_train = torch.tensor(x_train,requires_grad=True)

    yv1 = yyv[iy,:][:,ix].reshape(-1,1)[ind_unmask]
    # y_train = torch.from_numpy(yv1).float()
    # y_train = torch.tensor(y_train,requires_grad=True)
    # #%% For library : requires_grad=True
    # xx_torch = torch.from_numpy(xxv.reshape(-1,1)).float()
    # # xx_torch = torch.tensor(xx_torch,requires_grad=True)

    # yy_torch = torch.from_numpy(yyv.reshape(-1,1)).float()
    # # yy_torch = torch.tensor(yy_torch,requires_grad=True)
    #%%
    #return xx_torch, yy_torch, x_train, y_train, scale, ix, iy, ind_unmask
    # return xx_torch, yy_torch, xv1, yv1, scale, ix, iy, ind_unmask
    return 0, 0, xv1, yv1, scale, ix, iy, ind_unmask