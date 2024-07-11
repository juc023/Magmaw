from scipy.linalg import dft
from scipy.linalg import toeplitz
import os
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import torch.nn.functional as F
import math
import pdb
from . import networks
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import pylab
from matplotlib import pyplot as plt
from matplotlib.mlab import psd
PI = math.pi

class Soft_Constell_QAM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, constell, hardness):
        # x: batch, 1, packet, 64, 2
        # contesll: 16, 2
        min_idx = torch.norm(x.unsqueeze(4) - constell, dim=5).argmin(4)
        y = constell[min_idx]
        
        with torch.enable_grad():
            softmax = nn.Softmax(dim=4)
            softmaxed = softmax(-1 * hardness * torch.square(torch.norm(x.unsqueeze(4) - constell, dim=5, p=2)))
            softmaxed2 = torch.matmul(softmaxed, constell)
            ctx.save_for_backward(x, softmaxed2)
        return y 
    
    @staticmethod
    def backward(ctx, grad_output):
        x, softmaxed2 = ctx.saved_tensors
        grad_input = grad_num_samples = None
        
        grad_input, = torch.autograd.grad(softmaxed2, x, grad_outputs=grad_output)
        
        return grad_input, None, None


class BatchConv1DLayer(nn.Module):
    def __init__(self, stride=1,
                 padding=0, dilation=1):
        super(BatchConv1DLayer, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x, weight, bias=None):
        if bias is None:
            assert x.shape[0] == weight.shape[0], "dim=0 of x must be equal in size to dim=0 of weight"
        else:
            assert x.shape[0] == weight.shape[0] and bias.shape[0] == weight.shape[
                0], "dim=0 of bias must be equal in size to dim=0 of weight"

        b_i, b_j, c, h = x.shape
        b_i, out_channels, in_channels, kernel_width_size = weight.shape

        out = x.permute([1, 0, 2, 3]).contiguous().view(b_j, b_i * c, h)
        weight = weight.view(b_i * out_channels, in_channels, kernel_width_size)

        out = F.conv1d(out, weight=weight, bias=None, stride=self.stride, dilation=self.dilation, groups=b_i,
                       padding=self.padding)

        out = out.view(b_j, b_i, out_channels, out.shape[-1])

        out = out.permute([1, 0, 2, 3])

        if bias is not None:
            out = out + bias.unsqueeze(1).unsqueeze(3)

        return out

def Normalize(x, pwr=1):
    '''
    Normalization function
    '''
    power = torch.mean(x**2, (-2,-1), True)
    alpha = np.sqrt(pwr/2)/torch.sqrt(power)
    return alpha*x

def move_subcarriers(x):
    # x: [1, 1, 128, 64, 2]
    null_indices = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
    pilot_indices = [11, 25, 39, 53]
    active_indices = [i for i in range(64) if (i not in null_indices) and (i not in pilot_indices)]    

    x_active = x[:, :, :, active_indices, 2]

class Clipping(nn.Module):
    '''
    Simulating the Clipping effect
    ''' 
    def __init__(self, opt):
        super(Clipping, self).__init__()
        self.CR = opt.CR  # Clipping ratio
    	
    def forward(self, x):
        # Calculating the scale vector for each element  

        amp = torch.sqrt(torch.sum(x**2, -1, True))
        sigma = torch.sqrt(torch.mean(x**2, (-2,-1), True) * 2)
        ratio = sigma*self.CR/amp
        scale = torch.min(ratio, torch.ones_like(ratio))

        with torch.no_grad():
            bias = x*scale - x

        return x + bias

class Add_CP(nn.Module): 
    '''
    Add cyclic prefix 
    '''
    def __init__(self, opt):
        super(Add_CP, self).__init__()
        self.opt = opt
    def forward(self, x):
        '''
        x[...].size() = [64, 2]
        '''
        return torch.cat((x[...,-self.opt.K:,:], x), dim=-2)

class RM_CP(nn.Module):
    '''
    Remove cyclic prefix
    ''' 
    def __init__(self, opt):
        super(RM_CP, self).__init__()
        self.opt = opt
    def forward(self, x):
        return x[...,self.opt.K:, :]

class Add_CFO(nn.Module): 
    '''
    Simulating the CFO effect in baseband
    Ang: unit: (degree/sample)
    '''
    def __init__(self, opt):
        super(Add_CFO, self).__init__()
        self.opt = opt
    def forward(self, input, isVideo=None):
        # Input size:  NxPxSx(M+K)x2
        N = input.shape[0]     # Input batch size

        if self.opt.is_cfo_random:
            angs = (torch.rand(N)*2-1)*self.opt.max_ang
        else:
            angs = torch.ones(N)*self.opt.ang 

        if self.opt.is_trick:
            index = torch.arange(-self.opt.K, self.opt.M+self.opt.N_pilot).float()
            if isVideo == None:
                angs_all = torch.ger(angs, index).repeat((1,self.opt.img_S+1)).view(N, self.opt.img_S+1, self.opt.M+self.opt.N_pilot+self.opt.K)    # Nx(S+1)x(M+K)
            else:
                angs_all = torch.ger(angs, index).repeat((1,self.opt.video_S+1)).view(N, self.opt.video_S+1, self.opt.M+self.opt.N_pilot+self.opt.K)    # Nx(S+1)x(M+K)
        else:
            if isVideo == None:
                index = torch.arange(0, (self.opt.img_S+1)*(self.opt.M+self.opt.N_pilot+self.opt.K)).float()
                angs_all = torch.ger(angs, index).view(N, self.opt.img_S+1, self.opt.M+self.opt.N_pilot+self.opt.K)    # Nx(S+1)x(M+K)
            else:
                index = torch.arange(0, (self.opt.video_S+1)*(self.opt.M+self.opt.N_pilot+self.opt.K)).float()
                angs_all = torch.ger(angs, index).view(N, self.opt.video_S+1, self.opt.M+self.opt.N_pilot+self.opt.K)    # Nx(S+1)x(M+K)
            
        real = torch.cos(angs_all/360*2*PI).unsqueeze(1).unsqueeze(-1)   # Nx1xSx(M+K)x1 
        imag = torch.sin(angs_all/360*2*PI).unsqueeze(1).unsqueeze(-1)   # Nx1xSx(M+K)x1

        real_in = input[...,0].unsqueeze(-1)    # NxPx(Sx(M+K))x1 
        imag_in = input[...,1].unsqueeze(-1)    # NxPx(Sx(M+K))x1

        # Perform complex multiplication
        real_out = real*real_in - imag*imag_in
        imag_out = real*imag_in + imag*real_in

        return torch.cat((real_out, imag_out), dim=4) 

class Channel(nn.Module):
    '''
    Realization of passing multi-path channel
    '''
    def __init__(self, opt, device):
        super(Channel, self).__init__()

        # Assign the power delay spectrum
        self.opt = opt

        # Generate unit power profile
        power = torch.exp(-torch.arange(opt.L).float()/opt.decay).unsqueeze(0).unsqueeze(0).unsqueeze(3)  # 1x1xLx1
        self.power = power/torch.sum(power)
        self.device = device
        self.add_cp = Add_CP(opt)
        self.bconv1d = BatchConv1DLayer(padding=opt.L-1)  
        
        self.attack_csi = torch.load('models/csi_matrix.pt')
        
        self.attack_cof = torch.sqrt(self.power/2) * torch.randn(self.opt.gen_batch_size, self.opt.P, self.opt.L, 2)

    def get_attack_cof(self):
        return self.attack_cof

    def set_attack_cof(self, attack_cof):
        self.attack_cof = attack_cof
        self.random_attack_cof = attack_cof

    def change_attack_cof(self, attack_cof):
        self.random_attack_cof = torch.sqrt(self.power/2) * torch.randn(self.opt.gen_batch_size, self.opt.P, self.opt.L, 2)       # NxPxLx2

    def sample(self, N, P, M, L):
        # Sample the channel coefficients
        cof = torch.sqrt(self.power/2) * torch.randn(N, P, L, 2)
        cof_true = torch.cat((cof, torch.zeros((N,P,M-L,2))), 2)
        H_true = torch.fft(cof_true, 1)

        return cof, H_true

    def forward(self, input, cof=None, def_index=True):
        # Input size:   NxPx(S+1)(M+K)x2
        # Output size:  NxPx(L+(S+1)(M+K)-1)x2
        # Also return the true channel
        # Generate Channel Matrix

        N, P, SMK, _ = input.shape
        
        if cof is None:
            cof = torch.sqrt(self.power/2) * torch.randn(N, P, self.opt.L, 2)       # NxPxLx2

        cof_true = torch.cat((cof, torch.zeros((N,P,self.opt.M-self.opt.L,2))), 2) # NxPxMx2
        H_true = torch.fft(cof_true, 1)  # NxPxMx2

        signal_real = input[...,0].view(N*P, 1, 1, -1)       # (NxP)x((S+1)x(M+K))x1
        signal_imag = input[...,1].view(N*P, 1, 1, -1)       # (NxP)x((S+1)x(M+K))x1

        ind = torch.linspace(self.opt.L-1, 0, self.opt.L).long() #[7, 6, ... 0]

        cof_real = cof[...,0][...,ind].view(N*P, 1, 1, -1).to(self.device) 
        cof_imag = cof[...,1][...,ind].view(N*P, 1, 1, -1).to(self.device)
        # torch.Size([1, 1, 1, 52])
        
        output_real = self.bconv1d(signal_real, cof_real) - self.bconv1d(signal_imag, cof_imag)   # (NxP)x(L+(S+1)(M+K)-1)x1
        output_imag = self.bconv1d(signal_real, cof_imag) + self.bconv1d(signal_imag, cof_real)   # (NxP)x(L+(S+1)(M+K)-1)x1

        output = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1)   # (NxP)x(L+SMK-1)x2

        return output.view(N,P,self.opt.L + SMK - 1, 2), H_true

    def attack(self, input, cof=None, def_index=True):
        # Input size:   NxPx(S+1)(M+K)x2
        # Output size:  NxPx(L+(S+1)(M+K)-1)x2
        # Also return the true channel
        # Generate Channel Matrix

        N, P, SMK, _ = input.shape
        adv_csi = self.add_cp(self.attack_csi)
        
        signal_real = input[...,0].view(N*P, 1, -1, self.opt.M+self.opt.K)       # (NxP)x((S)x(M))x1
        signal_imag = input[...,1].view(N*P, 1, -1, self.opt.M+self.opt.K)       # (NxP)x((S)x(M))x1

        cof_real = adv_csi[...,0].view(N*P, 1, 1, -1).expand(N*P, 1, signal_real.size(2), self.opt.M+self.opt.K).to(self.device)
        cof_imag = adv_csi[...,1].view(N*P, 1, 1, -1).expand(N*P, 1, signal_imag.size(2), self.opt.M+self.opt.K).to(self.device)
        
        output_real = signal_real * cof_real - cof_real * cof_imag    # (NxP)x(L+(S+1)(M+K)-1)x1
        output_imag = signal_real * cof_imag + signal_imag * cof_real # (NxP)x(L+(S+1)(M+K)-1)x1

        # random frequency shift
        output_real, output_imag = self.random_phase_rotation(output_real, output_imag)

        output = torch.cat((output_real.view(N*P,-1,1), output_imag.view(N*P,-1,1)), -1)   # (NxP)x(L+SMK-1)x2

        return output.view(N*P,1,-1,2)
    
    def random_phase_rotation(self, real, imag):
        theta = torch.randn(1).to(self.device)
        theta_cos = torch.cos(theta).to(self.device)
        theta_sin = torch.sin(theta).to(self.device)
        
        rotated_real = (theta_cos * real) - (imag * theta_sin)
        rotated_img = (theta_cos * imag) + (real * theta_sin)
        
        return rotated_real, rotated_img
    
def complex_division(no, de):
    a = no[...,0]
    b = no[...,1]
    c = de[...,0]
    d = de[...,1]

    out_real = (a*c+b*d)/(c**2+d**2)
    out_imag = (b*c-a*d)/(c**2+d**2)

    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_multiplication(x1, x2):
    real1 = x1[...,0]
    imag1 = x1[...,1]
    real2 = x2[...,0]
    imag2 = x2[...,1]

    out_real = real1*real2 - imag1*imag2
    out_imag = real1*imag2 + imag1*real2

    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_conjugate(x):
    out_real = x[...,0]
    out_imag = -x[...,1]
    return torch.cat((out_real.unsqueeze(-1), out_imag.unsqueeze(-1)),-1)

def complex_amp(x):
    real = x[...,0]
    imag = x[...,1]
    return torch.sqrt(real**2 + imag**2).unsqueeze(-1)

def ZadoffChu(order, length, index=0):
    cf = length % 2
    n = np.arange(length)
    arg = np.pi*order*n*(n+cf+2*index)/length
    zado = np.exp(-1j*arg)
    zado_real = torch.from_numpy(zado.real).unsqueeze(-1).float()
    zado_imag = torch.from_numpy(zado.imag).unsqueeze(-1).float()
    return torch.cat((zado_real, zado_imag), 1)

def channel_equalization(H_est, Y, noise_pwr):
    # H_est: NxPx1xMx2
    # Y: NxPxSxMx2
    null_indices = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
    pilot_indices = [11, 25, 39, 53]
    active_indices = [i for i in range(64) if (i not in null_indices) and (i not in pilot_indices)]

    no = complex_multiplication(Y, complex_conjugate(H_est))
    de = complex_amp(H_est)**2 + noise_pwr.unsqueeze(-1) 
    return no/de

def channel_est(pilot_tx, pilot_rx, noise_pwr):
    # pilot_tx: NxPx1xMx2
    # pilot_rx: NxPxS'xMx2
    null_indices = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
    pilot_indices = [11, 25, 39, 53]
    active_indices = [i for i in range(64) if (i not in null_indices) and (i not in pilot_indices)]
    
    pilot_tx_48 = pilot_tx[active_indices, :]
    
    no = complex_multiplication(torch.mean(pilot_rx, 2, True), complex_conjugate(pilot_tx_48))
    de = 1+noise_pwr.unsqueeze(-1)/pilot_rx.shape[2]
    return no/de

class OFDM_channel(nn.Module):
    '''
    SImulating the end-to-end OFDM system with non-linearity
    '''
    def __init__(self, opt, device, pwr = 1):
        super(OFDM_channel, self).__init__()
        self.opt = opt
        self.device = device
        # Setup the add & remove CP layers
        self.add_cp = Add_CP(opt)
        self.rm_cp = RM_CP(opt)

        # Setup the channel layer
        self.channel = Channel(opt, device)
        self.clip = Clipping(opt)
 
        self.pilot = torch.load('models/lts_freq.pt')
        
        self.pilot = Normalize(self.pilot, pwr=pwr)
        self.pilot = self.pilot.to(device)
        self.pilot_cp = self.add_cp(torch.ifft(self.pilot, 1)).repeat(opt.P, opt.N_pilot,1,1)         #1xMx2  => PxS'x(M+K)x2

        self.pwr = pwr
        
        self.soft_hard_mod = networks.soft_to_hard_quantize(5)
        self.is_store_modulated_signal = False
        
        if self.opt.modulation != 'False':
            self.constell = torch.zeros(self.opt.m_degree,2).to(self.device)
            self.constell = self.create_modulation(self.constell, self.opt.modulation, self.opt.m_degree)
            self.demodulation = Soft_Constell_QAM
        else:
            self.constell = None

    def change_constellation(self):
        if self.opt.modulation != 'False':
            self.constell = torch.zeros(self.opt.m_degree,2).to(self.device)
            self.constell = self.create_modulation(self.constell, self.opt.modulation, self.opt.m_degree)
        else:
            self.constell = None        

    def print_modulated_signal(self, path, ref_path=None):
        self.is_store_modulated_signal = True
        self.path_modulated_signal = path
        self.ref_path_modulated_signal = ref_path

    def sample(self, N):
        return self.channel.sample(N, self.opt.P, self.opt.M, self.opt.L)

    def PAPR(self, x):
        power = torch.mean(x**2, (-2,-1))*2
        max_pwr, _ = torch.max(torch.sum(x**2, -1), -1)
        
        return max_pwr/power

    def random_time_shift(self, x):
        real = x[..., 0]
        imag = x[..., 1]
        
        freqs = np.fft.fftfreq(64, d=1/20e6)
        freqs_tensor = torch.tensor(freqs, dtype=torch.float32).to(self.device)

        tau = np.random.uniform(0, 1 / (20e6 / 64)) * 1e6 
        tau_seconds = tau * 1e-6
        theta = -2 * np.pi * freqs_tensor * tau_seconds
        
        theta_cos = torch.cos(theta).to(self.device)
        theta_sin = torch.sin(theta).to(self.device)
        
        rotated_real = (theta_cos * real) - (imag * theta_sin)
        rotated_img = (theta_cos * imag) + (real * theta_sin)
        
        x_shift_real = rotated_real.unsqueeze(-1)
        x_shift_imag = rotated_img.unsqueeze(-1)
        x_shift_freq = torch.cat((x_shift_real, x_shift_imag), -1)
        
        return x_shift_freq

    def forward(self, x, SNR, size_latent, perturbation=None, cof=None):
        # Input size: NxPxSxMx2   The information to be transmitted
        # cof denotes given channel coefficients

        N = x.shape[0]

        # Modulation
        if ((self.opt.modulation == 'QAM') | (self.opt.modulation == 'PSK')):
            x = Soft_Constell_QAM.apply(x, self.constell, 5)

        # Normalize the input power in frequency domain
        norm_x = Normalize(x, pwr=self.pwr)

        # IFFT:                    NxPxSxMx2  => NxPxSxMx2
        x = torch.ifft(norm_x, 1)

        # Add Cyclic Prefix:       NxPxSxMx2  => NxPxSx(K+M)x2
        x = self.add_cp(x)

        # Reshape:
        # x: NxPxSx(K+M)x2 => NxPx(Sx(K+M))x2
        # pilot: N x P x (N_pilot x (K+M)) x 2

        x = x.view(N, self.opt.P, size_latent*(self.opt.M+self.opt.K), 2)
        pilot = self.pilot_cp.repeat(N,1,1,1,1).view(N, self.opt.P, self.opt.N_pilot*(self.opt.M+self.opt.K), 2)            
        
        # Signal clipping (optional)       
        if self.opt.is_clip:
            with torch.no_grad():
                pwr_pre = torch.mean(x**2, (-2,-1), True) * 2
            x = self.clip(x)
            with torch.no_grad():
                pwr = torch.mean(x**2, (-2,-1), True) * 2
                alpha = torch.sqrt(pwr_pre/2)/torch.sqrt(pwr/2)
            x = alpha*x            
        PAPR = self.PAPR(x)

        # Add pilot:               NxPxSx(M+K)x2  => NxPx(S+1)x(M+K)x2
        x = torch.cat((pilot, x), 2)

        # Pass the Channel:        NxPx(S+1)(M+K)x2  =>  NxPx((S+1)(M+K)+L-1)x2
        y, H_true = self.channel(x, cof=cof)

        # Calculate the power of received signal
        # 'ins': instantaeous noise calculated at the receiver
        # 'avg': average noise calculated at the transmitter
        with torch.no_grad(): 
            if self.opt.SNR_cal == 'ins':    
                pwr = torch.mean(y**2, (-2,-1), True) * 2
                noise_pwr = pwr*10**(-SNR/10)
            elif self.opt.SNR_cal == 'avg':
                pwr = torch.mean(y**2, (-2,-1), True) * 2
                noise_pwr = self.pwr*10**(-SNR/10)/self.opt.M
                noise_pwr = noise_pwr * torch.ones_like(pwr)

        # Generate random noise
        noise = torch.sqrt(noise_pwr/2) * torch.randn_like(y)

        # Attack        
        if self.opt.is_attack == 'black':
            # time domain shift corresponds to phase rotation
            perturbation = self.random_time_shift(perturbation)
            
            pert = torch.ifft(perturbation, 1)

            # OFDM shuffling
            ofdm_shuffle = torch.randperm(64)
            shuffled_pert = pert[:,:,:,ofdm_shuffle,:]

            # add cp
            shuffled_pert = self.add_cp(shuffled_pert)

            shuffled_pert = shuffled_pert.view(N, -1, (size_latent+self.opt.N_pilot)*(self.opt.M+self.opt.K), 2)
            
            max_pert_pwr = pwr*10**(self.opt.psr/10)
            max_pert_pwr = torch.tensor([[[[max_pert_pwr]]]], requires_grad=True).to(self.device)
            pert_pwr = torch.mean(shuffled_pert**2, (-2,-1), True) * 2

            float_max_pert_pwr = max_pert_pwr.detach().cpu().squeeze().item()
            float_pert_pwr = pert_pwr.mean().detach().cpu().squeeze().item()
            
            if float_pert_pwr > float_max_pert_pwr:
                normalized_pert = torch.sqrt(max_pert_pwr) / torch.sqrt(pert_pwr) * shuffled_pert
            else:
                normalized_pert = shuffled_pert

            normalized_pert_pwr = torch.mean(normalized_pert**2, (-2,-1), True) * 2

            # Attack Channel
            pert_noisy = self.channel.attack(normalized_pert, cof=cof)

            y_noisy = y[:,:,:(size_latent+self.opt.N_pilot)*(self.opt.M+self.opt.K),:] + noise[:,:,:(size_latent+self.opt.N_pilot)*(self.opt.M+self.opt.K),:] + pert_noisy
                
        elif self.opt.is_attack == 'random':
            pert = torch.ifft(perturbation, 1)

            pert = pert.view(N, self.opt.P, (size_latent+self.opt.N_pilot)*(self.opt.M+self.opt.K), 2)
            
            max_pert_pwr = pwr*10**(self.opt.psr/10)
            pert_pwr = torch.mean(pert**2, (-2,-1), True) * 2
            
            float_max_pert_pwr = max_pert_pwr.detach().cpu().squeeze().item()
            float_pert_pwr = pert_pwr.mean().detach().cpu().squeeze().item()
            
            if float_pert_pwr > float_max_pert_pwr:
                random_pert = torch.sqrt(max_pert_pwr/2) / torch.sqrt(pert_pwr) * pert
            else:
                random_pert = pert

            pert_noisy, pert_H_true = self.channel(random_pert, cof=cof)

            y_noisy = y + noise + pert_noisy
            normalized_pert_pwr = torch.mean(random_pert**2, (-2,-1), True) * 2
        else:
            y_noisy = y + noise
            normalized_pert_pwr = None
            max_pert_pwr = None
        
        output = \
            y_noisy[:,:,:(size_latent+self.opt.N_pilot)*(self.opt.M+self.opt.K),:].view(N, self.opt.P, size_latent+self.opt.N_pilot, self.opt.M+self.opt.K, 2)            
        
        y_pilot = output[:,:,:self.opt.N_pilot,:,:]         # NxPxS'x(M+K)x2
        y_sig = output[:,:,self.opt.N_pilot:,:,:]           # NxPxSx(M+K)x2

        # Remove Cyclic Prefix":   
        info_pilot = self.rm_cp(y_pilot)    # NxPxS'xMx2
        info_sig = self.rm_cp(y_sig)        # NxPxSxMx2

        # FFT:                     
        info_pilot = torch.fft(info_pilot, 1)
        info_sig = torch.fft(info_sig, 1)

        return info_pilot, info_sig, H_true, noise_pwr, PAPR, normalized_pert_pwr, pwr

    def sdr_receiver(self):
        # Convert Index to Complex Value
        if ((self.opt.modulation == 'QAM') | (self.opt.modulation == 'PSK')):
            received_constell_idx = np.loadtxt(self.path_modulated_signal).reshape((1, 1, -1, 64))
            received_x = self.SDR_Constell_Remapping(received_constell_idx, self.constell).unsqueeze(0)
        else:
            raise Exception("Please set modulation for decoding SDR results")

        # Normalize the input power in frequency domain
        received_x = Normalize(received_x, pwr=self.pwr)

        return received_x

    def SDR_Constell_Remapping(self, received_idx, constell):
        y = constell[received_idx]
        
        return y

    def create_modulation(self, constell, Mod, M, P=1):
        index = [i for i in range(M)]

        if Mod == 'PSK':
            if M == 4:
                phi = np.pi/4 # phase rotation
                s = list(np.exp(1j * phi + 1j * 2 * np.pi * np.array(index) / M))
        else:
            c = np.sqrt(M)
            b = -2 * (np.array(index) % c) + c - 1
            a = 2 * np.floor(np.array(index) / c) - c + 1
            s = list((a + 1j * b))

        for i in range(M):
            constell[i][0] = s[i].real / np.sqrt(M) 
            constell[i][1] = s[i].imag / np.sqrt(M) 

        return constell

    def dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()


    def bin2dec(self, b, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)
