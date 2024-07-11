import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1
from .entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
#from .utils.stream_helper import get_downsampled_shape
from .layers.layers import MaskedConv2d, subpel_conv3x3
from torch.autograd import Variable
from .base_model import BaseModel
from . import channel
import pdb
import os
import numpy as np

class VideoJSCCOFDMModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        out_channel_mv = 64
        out_channel_M = opt.video_C_channel #96
        out_channel_N = 64

        self.perturbation = None

        self.out_channel_mv = out_channel_mv
        self.out_channel_M = out_channel_M
        self.out_channel_N = out_channel_N

        self.tx_feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.tx_context_refine = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.rx_feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.rx_context_refine = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N+3, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.tx_contextualDecoder_part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.tx_contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N*2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.rx_contextualDecoder_part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.rx_contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N*2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.mv_Encoder = nn.Sequential(
            nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_M, 3, stride=2, padding=1),
        )

        self.mv_Decoder_part1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel_M, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, 2, 3, stride=2, padding=1, output_padding=1),
        )

        self.mv_Decoder_part2 = nn.Sequential(
            nn.Conv2d(5, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )

        self.opticFlow = ME_Spynet()

        self.opt = opt
        self.channel = channel.OFDM_channel(opt, self.device, pwr=1)

    def set_input(self, input_image, tx_referframe, rx_referframe):
        self.input = input_image.clone().to(self.device)
        self.tx_ref = tx_referframe.clone().to(self.device)
        self.rx_ref = rx_referframe.clone().to(self.device)

    def set_latent_size(self, latent_size):
        self.latent_size = latent_size

    def tx_context_generation(self, ref, mv):
        ref_feature =  self.tx_feature_extract(ref)
        prediction_init = flow_warp(ref_feature, mv)
        #print(prediction_init.size())
        context =  self.tx_context_refine(prediction_init)

        return context
    
    def rx_context_generation(self, ref, mv):
        ref_feature =  self.tx_feature_extract(ref)
        prediction_init = flow_warp(ref_feature, mv)
        #print(prediction_init.size())
        context =  self.tx_context_refine(prediction_init)

        return context

    def mv_refine(self, ref, mv):
        return self.mv_Decoder_part2(torch.cat((mv, ref), 1)) + mv

    def set_optimizer_perturbation(self, wireless_perturbation):
        self.perturbation = wireless_perturbation
        self.optimizer_P = torch.optim.Adam([self.perturbation], lr=self.opt.attack_lr)
        
    def initialize_perturbation(self):
        self.perturbation = None
        
    def set_blackbox_perturbation(self, wireless_perturbation):
        N = wireless_perturbation.size(0)
        
        self.opt.pert_latent_size = int(wireless_perturbation.size(1)*wireless_perturbation.size(2)*wireless_perturbation.size(3) / (128 * self.opt.P))
        
        wireless_perturbation = \
            wireless_perturbation.view(self.opt.gen_batch_size, self.opt.P, self.opt.pert_latent_size, 2, self.opt.M).permute(0,1,2,4,3) 

        total_elements = N * self.opt.P * self.opt.video_S * self.opt.M *2
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

        total_elements = self.opt.P * self.opt.video_S * self.opt.M
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

    def backward_generator(self):
        self.forward()
        self.criterionL2 = torch.nn.MSELoss()
        self.loss_P_L2 = -self.criterionL2(self.rx_clipped_recon_image, self.real_B)


    def set_random_perturbation(self, wireless_perturbation):
        self.perturbation = wireless_perturbation

    def transmitter(self):

        m_t = self.opticFlow(self.input, self.tx_ref)
        y_mv = self.mv_Encoder(m_t)

        compressed_y_mv = y_mv

        quant_m_t = self.mv_Decoder_part1(compressed_y_mv)
        quant_m_t_refine = self.mv_refine(self.tx_ref, quant_m_t)
        context = self.tx_context_generation(self.tx_ref, quant_m_t_refine)

        feature = self.contextualEncoder(torch.cat((self.input, context), dim=1))
        feature_renorm = feature

        compressed_y_renorm = feature_renorm

        recon_image_feature = self.tx_contextualDecoder_part1(compressed_y_renorm)
        recon_image = self.tx_contextualDecoder_part2(torch.cat((recon_image_feature, context) , dim=1))

        return recon_image, compressed_y_mv, compressed_y_renorm

    def receiver(self, compressed_y_mv, compressed_y_renorm):
        quant_m_t = self.mv_Decoder_part1(compressed_y_mv)
        quant_m_t_refine = self.mv_refine(self.rx_ref, quant_m_t)  
        context = self.tx_context_generation(self.rx_ref, quant_m_t_refine)

        recon_image_feature = self.rx_contextualDecoder_part1(compressed_y_renorm)
        recon_image = self.rx_contextualDecoder_part2(torch.cat((recon_image_feature, context) , dim=1))

        return recon_image

    def channel_estimation(self, out_pilot, noise_pwr):
        return channel.channel_est(self.channel.pilot, out_pilot, self.opt.M*noise_pwr)

    def equalization(self, H_est, out_sig, noise_pwr):
        return channel.channel_equalization(H_est, out_sig, self.opt.M*noise_pwr)

    def forward(self, cof_in=None):

        N = self.input.shape[0]

        self.real_B = self.input

        # Generate information about the channel when available
        if cof_in is not None:
            cof, H_true = cof_in
        elif cof_in is None and self.opt.feedforward == 'OFDM-feedback':
            cof, H_true = self.channel.sample(N)
        else:
            cof, H_true = None, None

        # tx
        tx_recon_image, compressed_y_mv, compressed_y_renorm = \
            self.transmitter()

        if (self.channel.is_store_modulated_signal == True):
            np.savetxt(self.channel.ref_path_modulated_signal, tx_recon_image.reshape(-1).cpu().numpy(), fmt='%1.7f', newline='\n')

        # Reshape the latents to be transmitted
        # [batch, pilot, Number of packets, 2, Number of subcarriers per symbol] ->
        # [batch, pilot, Number of packets, Number of subcarriers per symbol, 2]
        # [B, C, 16, 16] -> [B, 1, C*16*16/64/2, 64, 2]
        self.tx_y = compressed_y_renorm.view(N, self.opt.P, self.opt.video_S, 2, self.opt.M).permute(0,1,2,4,3)
        self.tx_y_mv = compressed_y_mv.view(N, self.opt.P, self.opt.video_S, 2, self.opt.M).permute(0,1,2,4,3)
        

        # carrier allocation
        # 48 carriers: data, 
        # 4 carriers: pilot, 
        # 12 carries: NULL
        self.tx2_y, _ = self.padd_tensor(self.tx_y, N)
        self.tx2_y_mv, padd_num = self.padd_tensor(self.tx_y_mv, N)
        self.tx2_y = self.tx2_y
        self.tx2_y_mv = self.tx2_y_mv
        # torch.Size([1, 1, 32, 2, 64])
        self.tx2 = torch.cat((self.tx2_y, self.tx2_y_mv), 2).permute(0,1,2,4,3)
        
        self.latent_size = self.tx2.size(2)

        # channel
        out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR_y, normalized_pert_pwr, out_pwr = self.channel(self.tx2, SNR=self.opt.SNR, 
                                            size_latent= self.latent_size, 
                                            perturbation=self.perturbation, 
                                            cof=cof)
        self.H_true = self.H_true.to(self.device).unsqueeze(2)
        self.normalized_pert_pwr = normalized_pert_pwr
        self.out_pwr = out_pwr
        
        N, C, H, W = compressed_y_renorm.shape
         
        # carrier deallocation
        # 48 carriers: data, 
        # 4 carriers: pilot, 
        # 12 carries: NULL
        
        out_sig_y = out_sig[:,:,:self.tx2_y_mv.size(2),:,:]
        out_sig_y_mv = out_sig[:,:,self.tx2_y_mv.size(2):,:,:]

        out_sig_y, _ = self.remove_padd_tensor(out_sig_y, out_pilot, N, padd_num)
        out_sig_y_mv, x_pilot_48 = self.remove_padd_tensor(out_sig_y_mv, out_pilot, N, padd_num)
        #rx = torch.cat((out_sig_y, out_sig_y_mv), 2)

        # channel equalization
        self.H_est = self.channel_estimation(x_pilot_48, noise_pwr)
        rx_y = self.equalization(self.H_est, out_sig_y, noise_pwr) # [N, 1, 2*S, 64, 2]
        rx_y_mv = self.equalization(self.H_est, out_sig_y_mv, noise_pwr) # [N, 1, 2*S, 64, 2]
        rx_y = rx_y[:,:,:self.tx2_y_mv.size(2),:,:]
        rx_y_mv = rx_y_mv[:,:,:self.tx2_y_mv.size(2),:,:]

        rx_y = rx_y.reshape(N, self.opt.P, -1, 2)
        rx_y_mv = rx_y_mv.reshape(N, self.opt.P, -1, 2)
        if padd_num != 0:
            rx_y = rx_y[:, :, :-padd_num, :].view(N, self.opt.P, -1, 64, 2)
            rx_y_mv = rx_y_mv[:, :, :-padd_num, :].view(N, self.opt.P, -1, 64, 2)
        else:
            rx_y = rx_y.view(N, self.opt.P, -1, 64, 2)
            rx_y_mv = rx_y_mv.view(N, self.opt.P, -1, 64, 2)

        rx_y = self.channel.demodulation.apply(rx_y, self.channel.constell, 5)
        rx_y_mv = self.channel.demodulation.apply(rx_y_mv, self.channel.constell, 5)
        
        decompressed_y = rx_y.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W) #[N, 12, 8, 8]
        decompressed_y_mv = rx_y_mv.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W) #[N, 12, 8, 8]

        rx_recon_image = self.receiver(decompressed_y_mv, decompressed_y)

        #loss
        self.tx_mse_loss = torch.mean((tx_recon_image - self.input).pow(2))
        self.rx_mse_loss = torch.mean((rx_recon_image - self.input).pow(2))

        self.tx_clipped_recon_image = tx_recon_image.clamp(0., 1.)
        self.rx_clipped_recon_image = rx_recon_image.clamp(0., 1.)

    def backward_P(self):
        self.forward()
        self.criterionL2 = torch.nn.MSELoss()
        self.loss_P_L2 = -self.criterionL2(self.rx_clipped_recon_image, self.real_B)

        self.loss_P = self.loss_P_L2 
        self.optimizer_P.zero_grad()
        self.loss_P.backward()

        return self.perturbation, self.loss_P

    def load_model(self, model, cfg):
        load_suffix = 'iter_%d' % cfg.load_iter if cfg.load_iter > 0 else cfg.epoch
        load_filename = '%s_net.pth' % (load_suffix)
        save_dir = os.path.join(cfg.video_checkpoints_dir, cfg.video_name)
        load_path = os.path.join(save_dir, load_filename)
        
        print('loading the model from %s' % load_path)

        with open(load_path, 'rb') as f:
            pretrained_dict = torch.load(f)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        f = str(f)
        if f.find('iter') != -1 and f.find('.model') != -1:
            st = f.find('iter') + 4
            ed = f.find('.model', st)
            return int(f[st:ed])
        else:
            return 0

    def save_model(self, model, checkpoints_dir, iter):
        #save_filename = 'iter{}.model'.format(iter)
        save_filename = '%s_net.pth' % (iter)
        save_path = os.path.join(checkpoints_dir, save_filename)
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        torch.save(model.state_dict(), save_path)
        
def build_model():
        input_image = Variable(torch.zeros([4, 3, 256, 256]))
        tx_ref_image = Variable(torch.zeros([4, 3, 256, 256]))
        rx_ref_image = Variable(torch.zeros([4, 3, 256, 256]))

        DVST_model = VideoJSCCOFDMModel()
        DVST_model.eval()
        tx_recon_image, rx_recon_image = DVST_model(input_image, tx_ref_image, rx_ref_image)

        print(tx_recon_image.size())
        print(rx_recon_image.size())

