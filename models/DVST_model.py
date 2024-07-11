import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1
from entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from utils.stream_helper import get_downsampled_shape
from layers.layers import MaskedConv2d, subpel_conv3x3
from torch.autograd import Variable
import pdb

class DVST_net(nn.Module):
    def __init__(self):
        super().__init__()
        out_channel_mv = 128
        out_channel_M = 96
        out_channel_N = 64

        self.out_channel_mv = out_channel_mv
        self.out_channel_M = out_channel_M
        self.out_channel_N = out_channel_N

        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_z_mv = BitEstimator(out_channel_N)

        self.gaussian_encoder = GaussianEncoder()

        self.tx_feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.tx_feature_precoding = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.tx_context_refine_1 = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.tx_context_refine_2 = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.rx_feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.rx_feature_precoding = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.rx_context_refine_1 = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.rx_context_refine_2 = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.analysis_transform = nn.Sequential(
            nn.Conv2d(out_channel_M+3, out_channel_M, 3, stride=2, padding=1),
            GDN(out_channel_M),
            ResBlock(out_channel_M, out_channel_M, 3),
            nn.Conv2d(out_channel_M, out_channel_M, 3, stride=2, padding=1),
            GDN(out_channel_M),
            ResBlock(out_channel_M, out_channel_M, 3),
            nn.Conv2d(out_channel_M, out_channel_M, 3, stride=2, padding=1),
            GDN(out_channel_M),
            ResBlock(out_channel_M, out_channel_M, 3),
            nn.Conv2d(out_channel_M, out_channel_M, 3, stride=2, padding=1),
        )

        self.synthesis_transform_part1 = nn.Sequential(
            GDN(out_channel_M, inverse=True),
            ResBlock(out_channel_M, out_channel_M, 3),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_M, inverse=True),
            ResBlock(out_channel_M, out_channel_M, 3),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_M, inverse=True),
            ResBlock(out_channel_M, out_channel_M, 3),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 3,
                               stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 3,
                               stride=2, padding=1, output_padding=1),
        )
        
        self.synthesis_transform_part2 = nn.Sequential(
            ResBlock(out_channel_M*2, out_channel_M, 3),
            nn.Conv2d(out_channel_M, 3, 3, stride=1, padding=1),
        )

        self.mv_analysis_transform = nn.Sequential(
            nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            ResBlock(out_channel_mv, out_channel_mv, 3),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            ResBlock(out_channel_mv, out_channel_mv, 3),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            ResBlock(out_channel_mv, out_channel_mv, 3),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        )

        self.mv_synthesis_transform = nn.Sequential(
            GDN(out_channel_mv, inverse=True),
            ResBlock(out_channel_mv, out_channel_mv, 3),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            ResBlock(out_channel_mv, out_channel_mv, 3),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            ResBlock(out_channel_mv, out_channel_mv, 3),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            ResBlock(out_channel_mv*2, out_channel_mv, 3),
            nn.Conv2d(out_channel_mv, 2, 3, stride=1, padding=1),
        )

        self.priorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        self.priorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_N, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N, out_channel_N, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N, out_channel_M, 3, stride=1, padding=1)
        )

        self.mv_priorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_mv, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        self.mv_priorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_N, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N, out_channel_N * 3 // 2, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N * 3 // 2, out_channel_mv*2, 3, stride=1, padding=1)
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )

        self.auto_regressive = MaskedConv2d(
            out_channel_M, 2 * out_channel_M, kernel_size=5, padding=2, stride=1
        )

        self.mv_auto_regressive = MaskedConv2d(
            out_channel_mv, 2 * out_channel_mv, kernel_size=5, padding=2, stride=1
        )

        self.mv_entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_mv * 12 // 3, out_channel_mv * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 10 // 3, out_channel_mv * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 8 // 3, out_channel_mv * 6 // 3, 1),
        )

        self.temporalPriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.opticFlow = ME_Spynet()

    def tx_context_generation1(self, ref, mv):
        ref_feature =  self.tx_feature_extract(ref)
        prediction_init = flow_warp(ref_feature, mv)
        context =  self.tx_context_refine_1(prediction_init)

        return context

    def tx_context_generation2(self, ref, mv):
        ref_feature =  self.tx_feature_precoding(ref)
        prediction_init = flow_warp(ref_feature, F.avg_pool2d(
                mv, kernel_size=16, stride=16))
        context =  self.tx_context_refine_2(prediction_init)

        return context
    
    def rx_context_generation1(self, ref, mv):
        ref_feature =  self.tx_feature_extract(ref)
        prediction_init = flow_warp(ref_feature, mv)
        context =  self.tx_context_refine_1(prediction_init)

        return context

    def rx_context_generation2(self, ref, mv):
        ref_feature =  self.tx_feature_precoding(ref)
        prediction_init = flow_warp(ref_feature, F.avg_pool2d(
                mv, kernel_size=16, stride=16))
        context =  self.tx_context_refine_2(prediction_init)

        return context

    def sender(self, referframe, input_image):
        quant_noise_y=None
        quant_noise_z=None
        quant_noise_y_mv=None
        quant_noise_z_mv=None

        m_t = self.opticFlow(input_image, referframe)
        y_mv = self.mv_analysis_transform(m_t)
        z_mv = self.mv_priorEncoder(y_mv)

        if self.training:
            compressed_z_mv = z_mv + quant_noise_z_mv
        else:
            compressed_z_mv = torch.round(z_mv)

        params_mv = self.mv_priorDecoder(compressed_z_mv)

        if self.training:
            compressed_y_mv = y_mv + quant_noise_y_mv
        else:
            compressed_y_mv = torch.round(y_mv)
        
        ctx_params_mv = self.mv_auto_regressive(compressed_y_mv)
        gaussian_params_mv = self.mv_entropy_parameters(torch.cat((params_mv, ctx_params_mv), dim=1))
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)

        return compressed_y_mv
    
    def receiver(self, referframe, input_image):
        quant_noise_y=None
        quant_noise_z=None
        quant_noise_y_mv=None
        quant_noise_z_mv=None


def build_model():
        input_image = Variable(torch.zeros([4, 3, 256, 256]))
        ref_image = Variable(torch.zeros([4, 3, 256, 256]))

        DVST_model = DVST_net()
        DVST_model.eval()
        feature = DVST_model.sender(ref_image, input_image)

        print(feature.size())


if __name__ == '__main__':
    build_model()