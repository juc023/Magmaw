# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import channel
from . import mutual_info
import scipy.io as sio
import random
import pdb
from matplotlib import pyplot as plt
from torch_pesq import PesqLoss

def Channel_Normalize(x, pwr=1):
    '''
    Normalization function
    '''
    power = torch.mean(x**2, (-2,-1), True)
    alpha = np.sqrt(pwr/2)/torch.sqrt(power)
    return alpha*x

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)

    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)

class TextJSCCOFDMModel(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.perturbation = None

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L2', 'G_H', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake', 'real_B']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.opt.gan_mode != 'none' and self.isTrain:
            self.model_names = ['E', 'G', 'D']
        else:  # during test time, only load G
            self.model_names = ['E', 'G']
        
        if self.opt.feedforward == 'OFDM-CE-sub-EQ-sub':
            self.model_names += ['CE', 'EQ']
            self.netCE = networks.define_S(dim=6, dim_out=2, dim_in = 32,
                                        norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
            self.netEQ = networks.define_S(dim=6, dim_out=2, dim_in = 32,
                                        norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
        elif self.opt.feedforward == 'OFDM-CE-sub-EQ':
            self.model_names += ['CE']
            self.netCE = networks.define_S(dim=6, dim_out=2, dim_in = 32,
                                        norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
        elif self.opt.feedforward == 'OFDM-feedback':
            self.model_names += ['EQ', 'P']
            self.netEQ = networks.define_S(dim=6, dim_out=2, dim_in = 32,
                                        norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
            self.netP = networks.define_S(dim=self.opt.img_C_channel+2, dim_out=self.opt.img_C_channel, dim_in = 64,
                                        norm=opt.norm_EG, init_type='kaiming', init_gain=0.02, gpu_ids=self.gpu_ids)
            
        # define networks (both generator and discriminator)
        self.netE = networks.define_Text_E(opt, self.opt.num_vocab, gpu_ids=self.gpu_ids)

        self.netG = networks.define_Text_G(opt, self.opt.num_vocab, gpu_ids=self.gpu_ids)

        #if self.isTrain and self.is_GAN:  # define a discriminator;
        if self.opt.gan_mode != 'none':
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D,
                                          opt.norm_D, 'kaiming', 0.02, self.gpu_ids)

        self.soft_hard_mod = networks.soft_to_hard_quantize(5)

        print('---------- Networks initialized -------------')

        # set loss functions and optimizers
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss()
            #self.crossentropy = torch.nn.CrossEntropyLoss(reduction = 'none')

            params = list(self.netE.parameters()) + list(self.netG.parameters()) #+ list(self.tx_netG.parameters()) 

            if self.opt.feedforward == 'OFDM-CE-sub-EQ':
                params+= list(self.netCE.parameters())
            elif self.opt.feedforward == 'OFDM-CE-sub-EQ-sub':
                params+= list(self.netCE.parameters())
                params+= list(self.netEQ.parameters())
            elif self.opt.feedforward == 'OFDM-feedback':
                params+= list(self.netEQ.parameters())
                params+= list(self.netP.parameters())
                
            self.optimizer_G = torch.optim.Adam(params, lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
            self.optimizers.append(self.optimizer_G)

            self.Mine = mutual_info.Mine()
            self.Mine = mutual_info.init_device(self.Mine, self.gpu_ids)
            self.optimizer_MI = torch.optim.Adam(self.Mine.parameters(), lr=1e-3)

            if self.opt.gan_mode != 'none':
                params = list(self.netD.parameters())
                self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
        else:
            self.criterionL2 = torch.nn.MSELoss()

        self.opt = opt
        self.channel = channel.OFDM_channel(opt, self.device, pwr=1)


    def name(self):
        return 'JSCCOFDM_Model'

    def set_input(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)
        self.trg = image.clone().to(self.device)

    def set_pad_idx(self, pad_idx):
        self.pad_idx = pad_idx
        
    def set_mask(self, image):
        self.real_mask = image.clone().to(self.device)

    def set_encode(self, image):
        self.real_A = image.clone().to(self.device)
        self.real_B = image.clone().to(self.device)

    def set_decode(self, latent):
        self.latent = self.normalize(latent.to(self.device),1)

    def set_img_path(self, path):
        self.image_paths = path

    def set_latent_size(self, latent_size):
        self.latent_size = latent_size
        
    def set_optimizer_perturbation(self, wireless_perturbation):
        self.perturbation = wireless_perturbation
        self.optimizer_P = torch.optim.Adam([self.perturbation], lr=self.opt.attack_lr)

    def initialize_perturbation(self):
        self.perturbation = None

    def set_blackbox_perturbation(self, wireless_perturbation, text_S):
        N = wireless_perturbation.size(0)
        self.opt.text_S = text_S
        
        self.opt.pert_latent_size = int(wireless_perturbation.size(1)*wireless_perturbation.size(2)*wireless_perturbation.size(3) / 128)
        
        wireless_perturbation = \
            wireless_perturbation.view(self.opt.gen_batch_size, self.opt.P, self.opt.pert_latent_size, 2, self.opt.M).permute(0,1,2,4,3) 

        total_elements = N * self.opt.P * self.opt.text_S * self.opt.M
        padd_num = 48 - (total_elements % 48)
        if (total_elements % 48) != 0:
            self.latent_size = (total_elements + padd_num) / 48
        else:
            self.latent_size = total_elements / 48

        iter = int((self.latent_size + self.opt.N_pilot) / self.opt.pert_latent_size) -1
        frac = int((self.latent_size + self.opt.N_pilot) % self.opt.pert_latent_size)
        self.perturbation = wireless_perturbation.clone()

        for i in range(iter):
            self.perturbation = torch.cat((self.perturbation, wireless_perturbation), 2)
        if (iter < 0) & (frac != 0):
            self.perturbation = self.perturbation[:,:,:frac,:,:]
        elif (iter >= 0) & (frac != 0):
            self.perturbation = torch.cat((self.perturbation, wireless_perturbation[:,:,:frac,:,:]), 2)
        else:
            self.perturbation = torch.cat((self.perturbation, wireless_perturbation[:,:,:self.opt.N_pilot,:,:]), 2)
        assert(self.perturbation.size(2) == (self.latent_size+self.opt.N_pilot))

    def padd_tensor(self, x, N):
        x = x.reshape(N, self.opt.P, -1, 2)

        total_elements = self.opt.P * self.opt.text_S * self.opt.M
        padd_num = 48 - (total_elements % 48)
        if (total_elements % 48) != 0:
            x = torch.cat((x, torch.zeros((x.size(0), x.size(1), padd_num, 2)).to(self.device)), dim=2)
        else:
            padd_num = 0
            
        x = x.view(N, self.opt.P, -1, 48, 2).permute(0,1,2,4,3)

        x_64 = torch.zeros((x.size(0), x.size(1), x.size(2), 2, 64), dtype=torch.float32).to(self.device)
        null_indices = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
        pilot_indices = [11, 25, 39, 53]
        active_indices = [i for i in range(64) if (i not in null_indices) and (i not in pilot_indices)]
        
        x_64[:, :, :, :, active_indices] = x
        pilot_tensor = torch.tensor([1, -1, 1, 1], dtype=torch.float32).view(1, 1, 1, 1, 4).expand(N,self.opt.P,x.size(2),2,4).to(self.device)
        x_64[:, :, :, :, pilot_indices] = pilot_tensor
        
        return x_64, padd_num

    def remove_padd_tensor(self, x, x_pilot, N, padd_num):
        x = x.permute(0,1,2,4,3)
        null_indices = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
        pilot_indices = [11, 25, 39, 53]
        active_indices = [i for i in range(64) if (i not in null_indices) and (i not in pilot_indices)]
        
        x2 = x[:, :, :, :, active_indices].permute(0,1,2,4,3)

        active_indices = [i for i in range(64) if (i not in null_indices) and (i not in pilot_indices)]
        x_pilot_48 = x_pilot[:, :, :, active_indices, :]
        
        return x2, x_pilot_48

    def set_random_perturbation(self, wireless_perturbation):
        self.perturbation = wireless_perturbation

    #'''    
    def forward(self, cof_in=None):

        N = self.real_A.shape[0]
        trg_inp = self.real_A[:, :-1]
        self.trg_real = self.real_A[:, 1:]
        
        src_mask, look_ahead_mask = self.create_masks(self.real_A, trg_inp, self.pad_idx)

        # Pass the image through the image encoder
        # [batch, channel, w, h] = [1, 31, 16]
        self.latent = self.netE(self.real_A, src_mask)
        N1 = self.latent.shape[1]
        N2 = self.latent.shape[2]
        remainder = (N1*N2) % 128
        if remainder != 0:
            padding_size = 128-int(remainder)
            padding = torch.zeros(N, padding_size).to(self.device)
            padded_latent = torch.cat((self.latent.view(N, -1), padding),1)
        else:
            padding_size = 0
            padded_latent = self.latent.view(N, -1)

        # Generate information about the channel when available
        if cof_in is not None:
            cof, H_true = cof_in
        elif cof_in is None and self.opt.feedforward == 'OFDM-feedback':
            cof, H_true = self.channel.sample(N)
        else:
            cof, H_true = None, None
        
        # Pre-coding process when the channel feedback is available
        if self.opt.feedforward == 'OFDM-feedback':
            H_true = H_true.permute(0, 1, 3, 2).contiguous().view(N, -1, self.latent.shape[2], self.latent.shape[3]).to(self.latent.device)
            weights = self.netP(torch.cat((H_true, self.latent), 1))
            self.latent = self.latent*weights

        # Reshape the latents to be transmitted
        # 128, 1, 6, 2, 64 = [batch, pilot, Number of packets, 2, Number of subcarriers per symbol]
        # latent.size() = [B, 12, 8, 8] -> [B, 1, 6, 64, 2]
        self.tx = padded_latent.view(N, self.opt.P, self.opt.text_S, 2, self.opt.M).permute(0,1,2,4,3)

        # carrier allocation
        # 48 carriers: data, 
        # 4 carriers: pilot, 
        # 12 carries: NULL
        self.tx2, padd_num = self.padd_tensor(self.tx, N)
        self.tx2 = self.tx2.permute(0,1,2,4,3)
        self.latent_size = self.tx2.size(2)

        # Transmit through the channel
        out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR, normalized_pert_pwr, out_pwr = self.channel(self.tx2, 
                                                                                        SNR=self.opt.SNR, 
                                                                                        size_latent=self.latent_size, 
                                                                                        perturbation=self.perturbation, 
                                                                                        cof=cof)
        self.H_true = self.H_true.to(self.device).unsqueeze(2)
        self.normalized_pert_pwr = normalized_pert_pwr
        self.out_pwr = out_pwr

        # carrier deallocation
        # 48 carriers: data, 
        # 4 carriers: pilot, 
        # 12 carries: NULL
        out_sig, x_pilot_48 = self.remove_padd_tensor(out_sig, out_pilot, N, padd_num)

        # channel equalization
        self.H_est = self.channel_estimation(x_pilot_48, noise_pwr)
        rx = self.equalization(self.H_est, out_sig, noise_pwr) # [N, 1, 6, 64, 2]
        rx2 = rx.reshape(N, self.opt.P, -1, 2)
        if padd_num != 0:
            rx2 = rx2[:, :, :-padd_num, :]
        rx2 = rx2.view(N, self.opt.P, self.opt.text_S, 64, 2)
        
        self.dec_in = rx2.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1)[:, :padded_latent.shape[1]-padding_size].view(N, N1, N2) #[N, 12, 8, 8]
        channel_dec_output = self.netG.module.channel_decoder(self.dec_in)
        dec_output = self.netG.module.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
        self.fake = self.netG.module.dense(dec_output)


    def train_mi(self, cof_in=None):
        N = self.real_A.shape[0]

        # Pass the image through the image encoder
        # [batch, seq_len] = [64, 31] -> [64, 31, 16]
        sents_mask = (self.real_A == self.pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(self.device)  # [batch, 1, seq_len]
        self.latent = self.netE(self.real_A, sents_mask)
        N1 = self.latent.shape[1]
        N2 = self.latent.shape[2]
        remainder = (N1*N2) % 128
        if remainder != 0:
            padding_size = 128-int(remainder)
            padding = torch.zeros(N, padding_size).to(self.device)
            padded_latent = torch.cat((self.latent.view(N, -1), padding),1)
        else:
            padding_size = 0
            padded_latent = self.latent.view(N, -1)
        
        # Generate information about the channel when available
        if cof_in is not None:
            cof, H_true = cof_in
        elif cof_in is None and self.opt.feedforward == 'OFDM-feedback':
            cof, H_true = self.channel.sample(N)
        else:
            cof, H_true = None, None
        
        # Pre-coding process when the channel feedback is available
        if self.opt.feedforward == 'OFDM-feedback':
            H_true = H_true.permute(0, 1, 3, 2).contiguous().view(N, -1, self.latent.shape[2], self.latent.shape[3]).to(self.latent.device)
            weights = self.netP(torch.cat((H_true, self.latent), 1))
            self.latent = self.latent*weights

        # Reshape the latents to be transmitted
        # 128, 1, 6, 2, 64 = [batch, pilot, Number of packets, 2, Number of subcarriers per symbol]
        # latent.size() = [B, 12, 8, 8] -> [B, 1, 6, 64, 2]
        self.tx = padded_latent.view(N, self.opt.P, self.opt.text_S, 2, self.opt.M).permute(0,1,2,4,3)

        # carrier allocation
        # 48 carriers: data, 
        # 4 carriers: pilot, 
        # 12 carries: NULL
        self.tx2, padd_num = self.padd_tensor(self.tx, N)
        self.tx2 = self.tx2.permute(0,1,2,4,3)
        self.latent_size = self.tx2.size(2)

        # Transmit through the channel
        out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR, normalized_pert_pwr, max_pert_pwr = self.channel(self.tx2, SNR=self.opt.SNR, 
                                                                                        size_latent=self.latent_size, 
                                                                                        perturbation=self.perturbation, 
                                                                                        cof=cof)
        self.H_true = self.H_true.to(self.device).unsqueeze(2)
        self.normalized_pert_pwr = normalized_pert_pwr
        self.max_pert_pwr = max_pert_pwr

        # carrier deallocation
        # 48 carriers: data, 
        # 4 carriers: pilot, 
        # 12 carries: NULL
        out_sig, x_pilot_48 = self.remove_padd_tensor(out_sig, out_pilot, N, padd_num)

        self.H_est = self.channel_estimation(x_pilot_48, noise_pwr)
        self.H_est = self.H_est
        rx = self.equalization(self.H_est, out_sig, noise_pwr) # [N, 1, 6, 64, 2]
        rx2 = rx.reshape(N, self.opt.P, -1, 2)
        if padd_num != 0:
            rx2 = rx2[:, :, :-padd_num, :]
        rx2 = rx2.view(N, self.opt.P, self.opt.text_S, 64, 2)
        
        self.dec_in = rx2.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1)[:, :padded_latent.shape[1]-padding_size].view(N, N1, N2) #[N, 12, 8, 8]

    def create_masks(self, src, trg, padding_idx):

        src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]

        trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
        look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        
        return src_mask.to(self.device), combined_mask.to(self.device)

    def optimize_MI(self):
        self.Mine.train()
        self.optimizer_MI.zero_grad()
        self.train_mi()
        joint, marginal = mutual_info.sample_batch(self.latent, self.dec_in)
        mi_lb, _, _ = mutual_info.mutual_information(joint, marginal, self.Mine)
        loss_mine = -mi_lb

        loss_mine.backward()
        torch.nn.utils.clip_grad_norm_(self.Mine.parameters(), 10.0)
        self.optimizer_MI.step()  

        return loss_mine.item()

    def optimize_step(self):

        self.netE.train()
        self.netG.train()
        self.optimizer_G.zero_grad()
        self.forward()
        loss = self.calculate_CEloss(self.fake.contiguous().view(-1, self.fake.size(-1)), 
                                 self.trg_real.contiguous().view(-1))
        # MI
        if self.opt.enable_MI == True:
            self.Mine.eval()
            joint, marginal = mutual_info.sample_batch(self.latent, self.dec_in)
            mi_lb, _, _ = mutual_info.mutual_information(joint, marginal, self.Mine)
            loss_mine = -mi_lb
            loss = loss + 0.0009 * loss_mine
        
        loss.backward()
        self.optimizer_G.step()
        
        return loss.item()

    def calculate_CEloss(self, pred, trg):
        self.crossentropy = torch.nn.CrossEntropyLoss(reduction = 'none')
        loss = self.crossentropy(pred, trg)
        mask = (trg != self.pad_idx).type_as(loss.data)
        loss *= mask
        loss = loss.mean()
        
        return loss
    
    def calculate_Powerloss(self):
        pert = torch.ifft(self.perturbation, 1)
        pert = pert.view(self.opt.batch_size, self.opt.P, (self.latent_size+1)*(self.opt.M+self.opt.K), 2)
        pert_pwr = torch.mean(pert**2, (-2,-1), True) * 2
        
        return pert_pwr
    
    def Text_validate(self):
        self.forward()
        loss = self.calculate_CEloss(self.fake.contiguous().view(-1, self.fake.size(-1)), 
                                 self.trg_real.contiguous().view(-1))
        
        return loss.item()
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        _, pred_fake = self.netD(self.fake.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_data = self.real_B
        _, pred_real = self.netD(real_data)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        if self.opt.gan_mode in ['lsgan', 'vanilla']:
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        elif self.opt.gan_mode == 'wgangp':
            penalty, grad = networks.cal_gradient_penalty(self.netD, real_data, self.fake.detach(), self.device, type='mixed', constant=1.0, lambda_gp=10.0)
            self.loss_D = self.loss_D_fake + self.loss_D_real + penalty
            self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        if self.opt.gan_mode != 'none':
            feat_fake, pred_fake = self.netD(self.fake)
            self.loss_G_GAN = self.opt.lam_G * self.criterionGAN(pred_fake, True)
        else:
            self.loss_G_GAN = 0

        self.loss_G_L2 = self.criterionL2(self.fake, self.real_B) #* self.opt.lam_L1
        
        if self.opt.feedforward in ['OFDM-CE-sub-EQ', 'OFDM-CE-sub-EQ-sub']:
            self.loss_G_H = self.opt.lam_h * torch.mean((self.H_est-self.H_true)**2)*2  # Channel estimations are complex numbers
        else:
            self.loss_G_H = 0
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.loss_G_H # + self.loss_tx_G_L2 
        self.loss_G.backward()

    def backward_P(self):
        self.forward()
        self.crossentropy = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.loss_P = -self.calculate_CEloss(self.fake.contiguous().view(-1, self.fake.size(-1)), 
                                 self.trg_real.contiguous().view(-1))
        self.loss_pwr = self.calculate_Powerloss()
        self.loss = self.loss_P #+ self.loss_pwr * 1
        self.optimizer_P.zero_grad()
        self.loss.backward()

    def backward_generator(self):
        self.forward()
        self.crossentropy = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.loss_P_CE = -self.calculate_CEloss(self.fake.contiguous().view(-1, self.fake.size(-1)), 
                                 self.trg_real.contiguous().view(-1))

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        if self.opt.gan_mode != 'none':
            self.set_requires_grad(self.netD, True)   # enable backprop for D
            self.optimizer_D.zero_grad()              # set D's gradients to zero
            self.backward_D()                         # calculate gradients for D
            self.optimizer_D.step()                   # update D's weights
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        else:
            self.loss_D_fake = 0
            self.loss_D_real = 0
        # update G

        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(self.optimizer_G, 0.5)
        self.optimizer_G.step()             # udpate G's weights

    def get_channel(self):
        cof, _ = self.channel.sample()
        return cof


    def channel_estimation(self, out_pilot, noise_pwr):
        return channel.channel_est(self.channel.pilot, out_pilot, self.opt.M*noise_pwr)

    def equalization(self, H_est, out_sig, noise_pwr):
        return channel.channel_equalization(H_est, out_sig, self.opt.M*noise_pwr)
    
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
            constell[i][0] = s[i].real / np.sqrt(M) * 30
            constell[i][1] = s[i].imag / np.sqrt(M) * 30

        return constell

    def draw_constellation(self, tx):
        # [batch, pilot, Number of packets, Number of subcarriers per symbol, 2]

        x_list = []
        y_list = []

        num_points = 64
        
        for i in range(num_points):
            x_list.append(tx[0, 0, 0, i, 0].cpu().detach().numpy())
            y_list.append(tx[0, 0, 0, i, 1].cpu().detach().numpy())

        plt.scatter(x_list, y_list , color = 'cyan')
        plt.title('Constellation')
        plt.xlabel("Real")
        plt.ylabel("Imag")

        plt.savefig('constellation'
        )

            
    def greedy(self, start_symbol, max_len=30, cof_in=None):

        N = self.real_A.shape[0]
        src_mask = (self.real_A == self.pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(self.device) #[batch, 1, seq_len]

        # Pass the image through the image encoder
        self.latent = self.netE(self.real_A, src_mask)
        N1 = self.latent.shape[1]
        N2 = self.latent.shape[2]
        remainder = (N1*N2) % 128
        if remainder != 0:
            padding_size = 128-int(remainder)
            padding = torch.zeros(N, padding_size).to(self.device)
            padded_latent = torch.cat((self.latent.view(N, -1), padding),1)
        else:
            padding_size = 0
            padded_latent = self.latent.view(N, -1)

        # Generate information about the channel when available
        if cof_in is not None:
            cof, H_true = cof_in
        elif cof_in is None and self.opt.feedforward == 'OFDM-feedback':
            cof, H_true = self.channel.sample(N)
        else:
            cof, H_true = None, None
        
        # Pre-coding process when the channel feedback is available
        if self.opt.feedforward == 'OFDM-feedback':
            H_true = H_true.permute(0, 1, 3, 2).contiguous().view(N, -1, self.latent.shape[2], self.latent.shape[3]).to(self.latent.device)
            weights = self.netP(torch.cat((H_true, self.latent), 1))
            self.latent = self.latent*weights

        # Reshape the latents to be transmitted
        # 128, 1, 6, 2, 64 = [batch, pilot, Number of packets, 2, Number of subcarriers per symbol]
        # latent.size() = [B, 12, 8, 8] -> [B, 1, 6, 64, 2]
        self.tx = padded_latent.view(N, self.opt.P, self.opt.text_S, 2, self.opt.M).permute(0,1,2,4,3)

        # carrier allocation
        # 48 carriers: data, 
        # 4 carriers: pilot, 
        # 12 carries: NULL
        self.tx2, padd_num = self.padd_tensor(self.tx, N)
        self.tx2 = self.tx2.permute(0,1,2,4,3)
        self.latent_size = self.tx2.size(2)

        # Transmit through the channel
        out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR, normalized_pert_pwr, max_pert_pwr = self.channel(self.tx2, SNR=self.opt.SNR, 
                                                                                        size_latent=self.latent_size, 
                                                                                        perturbation=self.perturbation, 
                                                                                        cof=cof)
        self.H_true = self.H_true.to(self.device).unsqueeze(2)
        self.normalized_pert_pwr = normalized_pert_pwr
        self.max_pert_pwr = max_pert_pwr

        # carrier deallocation
        # 48 carriers: data, 
        # 4 carriers: pilot, 
        # 12 carries: NULL
        out_sig, x_pilot_48 = self.remove_padd_tensor(out_sig, out_pilot, N, padd_num)

        # channel equalization
        self.H_est = self.channel_estimation(x_pilot_48, noise_pwr)
        rx = self.equalization(self.H_est, out_sig, noise_pwr) # [N, 1, 6, 64, 2]
        rx2 = rx.reshape(N, self.opt.P, -1, 2)
        if padd_num != 0:
            rx2 = rx2[:, :, :-padd_num, :]
        rx2 = rx2.view(N, self.opt.P, self.opt.text_S, 64, 2)
        
        rx2 = self.channel.demodulation.apply(rx2, self.channel.constell, 5)
        
        self.dec_in = rx2.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1)[:, :padded_latent.shape[1]-padding_size].view(N, N1, N2) #[N, 12, 8, 8]

        memory = self.netG.module.channel_decoder(self.dec_in)
        outputs = torch.ones(self.real_A.size(0), 1).fill_(start_symbol).type_as(self.real_A.data)

        for i in range(max_len - 1):
            # create the decode mask
            trg_mask = (outputs == self.pad_idx).unsqueeze(-2).type(torch.FloatTensor) #[batch, 1, seq_len]
            look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
            combined_mask = torch.max(trg_mask, look_ahead_mask)
            combined_mask = combined_mask.to(self.device)

            # decode the received signal
            dec_output =  self.netG.module.decoder(outputs, memory, combined_mask, None)
            pred = self.netG.module.dense(dec_output)
            
            # predict the word
            prob = pred[: ,-1:, :]  # (batch_size, 1, vocab_size)

            # return the max-prob index
            _, next_word = torch.max(prob, dim = -1)
            
            #next_word = next_word.data[0]
            outputs = torch.cat([outputs, next_word], dim=1)
        return outputs
