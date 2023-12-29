#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 14:53:22 2021
Mar 21: fft_loss \ DOEST NOT WORK WITH SENSOR_RATIO!=1.0

@author: rmojgani
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
# Packages
try:
    from tqdm import tqdm
except:
    import os
    os.system('!pip install tqdm')
    from tqdm import tqdm
    
try:
    import rff
except:
    import os
    os.system('!pip install random-fourier-features-pytorch')
    import rff

class Net(nn.Module):
    # =========================================================================
    def __init__(self, hidden, DEEP, LAMBDA_REG=0.0, LOSS_FFT_CUT=40, RFF_SIGMA=0.0):
        
        # Packages
        try:
            from tqdm import tqdm
        except:
            import os
            os.system('!pip install tqdm')
            from tqdm import tqdm
           
        try:
            import rff
        except:
            import os
            os.system('!pip install random-fourier-features-pytorch')
            import rff

        # Variables
        self.hidden = hidden
        self.DEEP = DEEP
        # self.LR0 = LR0
        self.LAMBDA_REG = LAMBDA_REG
        self.LOSS_FFT_CUT = LOSS_FFT_CUT
        
        input  = 2 # x, y#, t
        output = 2 # psi1, psi2
        
        # Initialize
        super().__init__()
        # self.input = (nn.Linear(input, hidden))
        self.input = (nn.Linear(hidden, hidden))

        self.hidden1 = (nn.Linear(hidden, hidden))
        self.hidden2 = (nn.Linear(hidden, hidden))
        self.hidden3 = (nn.Linear(hidden, hidden))
        self.hidden4 = (nn.Linear(hidden, hidden))
        self.hidden5 = (nn.Linear(hidden, hidden))
        self.output = nn.Linear(hidden, output)
        
        self.encoding = rff.layers.GaussianEncoding(sigma=RFF_SIGMA, input_size=input, encoded_size=int(hidden/2) )

    # =========================================================================
    def forward(self, x, y):
        x00 = torch.cat((x,y), axis = 1)
        
        x0 = self.encoding(x00)

        x1 = torch.tanh(self.input(x0))# torch.tanh
        x2 = torch.tanh(self.hidden1(x1))
        x3 = torch.tanh(self.hidden2(x2))
        if self.DEEP == 3:
            u = (self.output(x3)).cuda()
        if self.DEEP == 4:
            x4 = torch.tanh(self.hidden3(x3))
            u = (self.output(x4)).cuda()
        if self.DEEP == 5:
            x4 = torch.tanh(self.hidden3(x3))
            x5 = torch.tanh(self.hidden4(x4))
            u = (self.output(x5)).cuda()
        if self.DEEP == 6:
            x4 = torch.tanh(self.hidden3(x3))
            x5 = torch.tanh(self.hidden4(x4))
            x6 = torch.tanh(self.hidden5(x5))
            u = (self.output(x6)).cuda()
        if self.DEEP == 10:
            x4 = torch.tanh(self.hidden3(x3))
            x5 = torch.tanh(self.hidden4(x4))
            x6 = torch.tanh(self.hidden5(x5))
            x7 = torch.tanh(self.hidden5(x6))
            x8 = torch.tanh(self.hidden5(x7))
            x9 = torch.tanh(self.hidden5(x8))
            x10 = torch.tanh(self.hidden5(x9))
            u = (self.output(x10)).cuda()
            
        return u
    # =========================================================================
    def lib(self, x, y, col=0, greedy_mem=True):
        output = self.forward(x, y)
        u = output[:,col].reshape(-1,1)
        
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_y = torch.autograd.grad(
            u, y, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        if greedy_mem:
            u = u.detach().cpu().numpy()[:,0]
                
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        if greedy_mem:
            u_xx = u_xx.detach().cpu().numpy()[:,0]
            
        u_xy = torch.autograd.grad(
            u_x, y, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        if greedy_mem:
            u_x = u_x.detach().cpu().numpy()[:,0]
            u_xy = u_xy.detach().cpu().numpy()[:,0]

        u_yx = torch.autograd.grad(
            u_y, x, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]

        if greedy_mem:
            u_yx = u_yx.detach().cpu().numpy()[:,0]

        u_yy = torch.autograd.grad(
            u_y, y, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        if greedy_mem:
            u_y = u_y.detach().cpu().numpy()[:,0]
            u_yy = u_yy.detach().cpu().numpy()[:,0]
            
        return [u, u_x, u_y, u_xx, u_yy, u_xy, u_yx]
    # =========================================================================
    def lib_DP(self, x, y, col=0, scale = {'x' : 1.0, 'y': 1.0}, greedy_mem=True):
        [u, u_x, u_y, u_xx, u_yy, u_xy, u_yx] = self.lib(x, y, col, greedy_mem)
        # ------------
        u_x = scale['x'] * u_x
        u_y = scale['y'] * u_y
        u_xx = (scale['x']**2) * u_xx
        u_yy = (scale['y']**2) * u_yy
        u_xy = (scale['x']*scale['y']) * u_xy
        u_yx = (scale['y']*scale['x']) * u_yx
        #u_xxx = (scale**3) * u_xxx
        #u_xxxx = (scale**4) * u_xxxx
        
        return [u, u_x, u_y, u_xx, u_yy, u_xy, u_yx]
    # =========================================================================
    def optimize_SGD(self, x, y, w, idx, NUM_EPOCHS_SGD, LR0):
        loss_fn = nn.MSELoss()

        optimizer = optim.SGD(self.parameters(), lr=LR0/10, momentum=0.9)

        step = 0
        # for epoch in range(0, NUM_EPOCHS_SGD):  # loop over the dataset multiple times
        for epoch in tqdm(range(0, NUM_EPOCHS_SGD)):
        
            # get the inputs; data is a list of [inputs, labels]
            x_batch, y_batch, w_batch = x[idx], y[idx], w[idx]
            
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            output = self.forward(x_batch.cuda(), y_batch.cuda())
            loss = loss_fn(output, w_batch.cuda())
            loss.backward()
            optimizer.step()
            
#            if loss.detach().to('cpu').numpy()>1e6:
#                print('Divergance .... ! loss = ', loss.detach().cpu().numpy() )
#                break 
            # print statistics
            if epoch % int(NUM_EPOCHS_SGD/10) == 0:    # print 10 times
                print('[%d, %5d] loss: %.3e' %
                      (epoch + 1, step + 1, loss))
                
                # End of warm-up
                if (epoch>10 and epoch < int(NUM_EPOCHS_SGD/9)):
                    print('End of warm-up period, LR updated to %.2e' %(LR0))
                    optimizer = optim.SGD(self.parameters(), lr=LR0, momentum=0.9)
    # =========================================================================
    def optimize_LBFGS(self, x, y, w, idx, NUM_EPOCHS_BFGS, LR0):
        """
        https://soham.dev/posts/linear-regression-pytorch/
        https://johaupt.github.io/python/pytorch/neural%20network/optimization/pytorch_lbfgs.html
        """
        optimizer = optim.LBFGS(self.parameters(), lr=LR0)
        loss_fn = nn.MSELoss()

        #for epoch in range(0, num_epochs):  # loop over the dataset multiple times
        for epoch in tqdm(range(0, NUM_EPOCHS_BFGS)):
        
            x_batch, y_batch, w_batch = x[idx], y[idx], w[idx]
        
            ### Add the closure function to calculate the gradient.
            def closure():
                if torch.is_grad_enabled():
                  optimizer.zero_grad()
                  
                output = self.forward(x_batch.cuda(), y_batch.cuda())
                
                loss = loss_fn(output, w_batch.cuda())
                
                if loss.requires_grad:
                  loss.backward(retain_graph=True)
              
                return loss
                    
            optimizer.step(closure)
        
            # calculate the loss again for monitoring
            self.forward(x_batch.cuda(), y_batch.cuda())
            loss = closure()
            
#            if loss.detach().to('cpu').numpy()>1e6:
#                print('Divergance .... ! loss = ', loss.detach().cpu().numpy() )
#                break 
            # print statistics
            if epoch % int(NUM_EPOCHS_BFGS/10) == 0:
                print('[%d] loss: %.3e' % (epoch + 1, loss))
                
    # =========================================================================
    def optimize_ADAM(self, x, y, w, idx, NUM_EPOCHS_ADAM, LR0, GAMMA_DIFF=0, GAMMA_LAPL=0):
    
        LAMBDA_REG = 0    
        # LAMBDA_REG = self.LAMBDA_REG

        #loss_fn = nn.MSELoss()
        
        optimizer = optim.Adam(self.parameters(), lr=LR0,\
                               betas=(0.9, 0.999), eps=1e-08, weight_decay=0,\
                               amsgrad=True)

        print('Train network: ADAM ...')
        tic = time.time()
        
        step = 0
        # for epoch in range(0, NUM_EPOCHS_ADAM):  # loop over the dataset multiple times
        for epoch in tqdm(range(0, NUM_EPOCHS_ADAM)):
        
            # get the inputs; data is a list of [inputs, labels]
#            x_batch, y_batch, w_batch = x[idx], y[idx], w[idx]
            x_batch, y_batch, w_batch = x, y, w
            
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            output = self.forward(x_batch.cuda(), y_batch.cuda())
            
            # loss = loss_fn(output, w_batch.cuda())
            loss = self.fft_loss(output, w_batch.cuda(),LAMBDA_REG)
            
            if GAMMA_DIFF!=0:
                loss += GAMMA_DIFF*loss_fn(output[:,0]-output[:,1], (w_batch[:,0]-w_batch[:,1]).cuda())
           
            if GAMMA_LAPL!=0:
                LAPL_psi1 = self.lapl(x_batch.cuda(), y_batch.cuda(), 0,  greedy_mem=False)
                LAPL_psi2 = self.lapl(x_batch.cuda(), y_batch.cuda(), 1,  greedy_mem=False)
                loss += GAMMA_LAPL*loss_fn(LAPL_psi1 + LAPL_psi2, 0*(w_batch[:,0]).cuda())

#            loss = loss1 + GAMMA_DIFF*loss2 + GAMMA_LAPL*loss3

            loss.backward()
            optimizer.step()
            
#            if loss.detach().to('cpu').numpy()>1e6:
#                print('Divergance .... ! loss = ', loss.detach().cpu().numpy() )
#                break 
            # print history
            # if epoch % int(NUM_EPOCHS_ADAM/10) == 0:    # print 10 times
            #     print('[%d, %5d] loss: %.3e,  %.3e,  %.3e,  %.3e' %
            #           (epoch + 1, step + 1, loss, loss1, loss2lay1, loss2lay2))
            #     if  epoch > int(NUM_EPOCHS_ADAM/2):
            #         LAMBDA_REG = self.LAMBDA_REG
            #         print('Adaptive lambda:', LAMBDA_REG )
                
        toc = time.time()
        print('ADAM: Elased time: %.2f min' %((toc-tic)/60) )
    # =========================================================================
#    def loss_diff(ouput, target):
#        # Find absolute difference
#        difF_ouput = ouput[:,0]-ouput[:,1]
#        difF_target = target[:,0]-target[:,1]
#
#        absolute_differences = np.absolute(differences)
#        # find the absolute mean
#        mean_absolute_error = absolute_differences.mean()
#        return mean_absolute_error
        # =========================================================================
    def lapl(self, x, y, col=0, greedy_mem=False):
        output = self.forward(x, y)
        u = output[:,col].reshape(-1,1)
        
        u_x = torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        u_y = torch.autograd.grad(
            u, y, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        if greedy_mem:
            u = u.detach().cpu().numpy()[:,0]
                
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        if greedy_mem:
            u_xx = u_xx.detach().cpu().numpy()[:,0]
            

        
        if greedy_mem:
            u_x = u_x.detach().cpu().numpy()[:,0]


        u_yy = torch.autograd.grad(
            u_y, y, 
            grad_outputs=torch.ones_like(x),
            retain_graph=True,
            create_graph=True
        )[0]
        
        if greedy_mem:
            u_y = u_y.detach().cpu().numpy()[:,0]
            u_yy = u_yy.detach().cpu().numpy()[:,0]
            
        return u_xx+u_yy
    # =========================================================================
    # def fft_loss(self, output, target):
    #     LAMBDA_REG = self.LAMBDA_REG
    #     LOSS_FFT_CUT = self.LOSS_FFT_CUT

    #     loss_fn = nn.MSELoss()
    #     loss_fft = nn.L1Loss()

    #     # loss1 = torch.mean((output-target)**2)
    #     loss1 = loss_fn(output, target)
        
    #     # out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=3)),dim=2)
    #     # target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=3)),dim=2) #nsamp , 2, 192,96
    #     # [192,96]
    #     # fft : 96
    #     # mean: 192
         
    #     target = target.reshape(192,96,2)
    #     output = output.reshape(192,96,2)
        
    #     out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=1)),dim=0)
    #     target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=1)),dim=0)


    #     # loss2 = torch.mean(torch.abs(out_fft[:,0,150:]-target_fft[:,0,150:]))
    #     # loss2 = loss_fn(out_fft[LOSS_FFT_CUT:,:], target_fft[LOSS_FFT_CUT:,:])
    #     loss2lay1 = loss_fft(out_fft[LOSS_FFT_CUT:,0], target_fft[LOSS_FFT_CUT:,0])
    #     loss2lay2 = loss_fft(out_fft[LOSS_FFT_CUT:,1], target_fft[LOSS_FFT_CUT:,1])
    #     loss2 = loss2lay1 + loss2lay2
        
    #     # import matplotlib.pylab as plt
    #     # plt.subplot(2,1,1)
    #     # plt.semilogy(out_fft[LOSS_FFT_CUT:,:].detach().cpu().numpy()[:,0]);
    #     # plt.semilogy(target_fft[LOSS_FFT_CUT:,:].detach().cpu().numpy()[:,0])
    #     # plt.subplot(2,1,2)
    #     # plt.semilogy(out_fft[LOSS_FFT_CUT:,:].detach().cpu().numpy()[:,1]);
    #     # plt.semilogy(target_fft[LOSS_FFT_CUT:,:].detach().cpu().numpy()[:,1])
        
    #     return (1.0-LAMBDA_REG)*loss1 + LAMBDA_REG*loss2
    # =========================================================================
    def fft_loss(self, output, target, LAMBDA_REG):
        #LAMBDA_REG = self.LAMBDA_REG
        #LOSS_FFT_CUT = self.LOSS_FFT_CUT
    
        loss_fn = nn.MSELoss()
        #loss_fft = nn.L1Loss()
        #loss_fft = nn.MSELoss()
        
        # loss1 = torch.mean((output-target)**2)
        loss1 = loss_fn(output, target)
        loss1 = loss1/loss_fn(target, 0*target)
        
        # out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=3)),dim=2)
        # target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=3)),dim=2) #nsamp , 2, 192,96
        # [192,96]
        # fft : 96
        # mean: 192
         
        #target = target.reshape(192,96,2)
        #output = output.reshape(192,96,2)
        
        #out_fft = torch.mean(torch.abs(torch.fft.rfft(output,dim=1)),dim=0)
        #target_fft = torch.mean(torch.abs(torch.fft.rfft(target,dim=1)),dim=0)
    
    
        # loss2 = torch.mean(torch.abs(out_fft[:,0,150:]-target_fft[:,0,150:]))
        # loss2 = loss_fn(out_fft[LOSS_FFT_CUT:,:], target_fft[LOSS_FFT_CUT:,:])
        #loss2lay1 = loss_fft(out_fft[LOSS_FFT_CUT:,0], target_fft[LOSS_FFT_CUT:,0])
        #loss2lay2 = loss_fft(out_fft[LOSS_FFT_CUT:,1], target_fft[LOSS_FFT_CUT:,1])
        
        #loss2lay1 = loss2lay1/loss_fft(target_fft[LOSS_FFT_CUT:,0],0*target_fft[LOSS_FFT_CUT:,0])
        #loss2lay2 = loss2lay2/loss_fft(target_fft[LOSS_FFT_CUT:,1],0*target_fft[LOSS_FFT_CUT:,1])
        #loss2 = loss2lay1 + loss2lay2
        
        # import matplotlib.pylab as plt
        # plt.figure()
        # plt.subplot(2,1,1)
        # plt.semilogy(out_fft[LOSS_FFT_CUT:,:].detach().cpu().numpy()[:,0]);
        # plt.semilogy(target_fft[LOSS_FFT_CUT:,:].detach().cpu().numpy()[:,0])
        # plt.subplot(2,1,2)
        # plt.semilogy(out_fft[LOSS_FFT_CUT:,:].detach().cpu().numpy()[:,1]);
        # plt.semilogy(target_fft[LOSS_FFT_CUT:,:].detach().cpu().numpy()[:,1])
        
        #loss = loss #(1.0-LAMBDA_REG)*loss1 + LAMBDA_REG*loss2
        
        return loss1 #, 0, 0, 0
