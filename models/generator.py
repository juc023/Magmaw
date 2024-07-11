import torch
from torch.nn import functional as F
import torch.nn as nn
from .layers.layers import subpel_conv3x3
from .video_net import GDN, ResBlock_LeakyReLU_0_Point_1, ResBlock
import pdb

class se_block_conv(nn.Module):
    def __init__(self, channel, kernel, stride, padding, enable):
        super(se_block_conv, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.enable = enable

        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding, bias=True)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding, bias=True)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        self.se_conv1 = nn.Conv2d(channel, channel//16, kernel_size=1)
        self.se_conv2 = nn.Conv2d(channel//16, channel, kernel_size=1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        output = F.relu(self.conv1_norm(self.conv1(x)))
        output = self.conv2_norm(self.conv2(output))
        
        if self.enable:
            se = F.avg_pool2d(output, output.size(2))
            se = F.relu(self.se_conv1(se))
            se = F.sigmoid(self.se_conv2(se))
            output = output * se

        output += x
        output = F.relu(output)
        return output

class se_block_deconv(nn.Module):
    def __init__(self, channel, kernel, stride, padding, enable):
        super(se_block_deconv, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding
        self.enable = enable

        self.conv1 = nn.ConvTranspose2d(channel, channel, kernel, stride, padding, bias=True)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.ConvTranspose2d(channel, channel, kernel, stride, padding, bias=True)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        self.se_conv1 = nn.Conv2d(channel, channel//16, kernel_size=1)
        self.se_conv2 = nn.Conv2d(channel//16, channel, kernel_size=1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        output = F.relu(self.conv1_norm(self.conv1(x)))
        output = self.conv2_norm(self.conv2(output))
        
        if self.enable:
            se = F.avg_pool2d(output, output.size(2))
            se = F.relu(self.se_conv1(se))
            se = F.sigmoid(self.se_conv2(se))
            output = output * se

        output += x
        output = F.relu(output)
        return output


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        try:
            m.bias.data.zero_()
        except:
            return

class Res_Generator(nn.Module):
    def __init__(self):
        super(Res_Generator, self).__init__()
        noise = 100
        channel = 64 #conf.channel_num
        block_num = 5
        enable_bias = False
        net_g_se = True

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(noise, channel*16, 4, 1, 0, bias=enable_bias),
            nn.InstanceNorm2d(channel*16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(channel*16, channel*4, 4, 2, 1, bias=enable_bias),
            nn.InstanceNorm2d(channel*4),
            nn.ReLU(inplace=True)
        )
        
        self.resnet_blocks = []
        for i in range(block_num):
                self.resnet_blocks.append(se_block_deconv(channel*4, 3, 1, 1, net_g_se))
                self.resnet_blocks[i].weight_init(0, 0.02)
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(channel*4, channel*1, 4, 2, 1, bias=enable_bias),
            nn.InstanceNorm2d(channel*1),
            nn.ReLU(inplace=True),

            nn.Conv2d(channel*1, 16, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        #pdb.set_trace()
        x = self.deconv1(x)
        #pdb.set_trace()
        x = self.resnet_blocks(x)
        #pdb.set_trace()
        x = self.deconv2(x)
        #pdb.set_trace()
        return x

class Generator(nn.Module):
    def __init__(self, opt) -> None:
        super(Generator, self).__init__()
        
        if opt.gen_channel_size == 32:
            self.main = nn.Sequential(
                # Input is 100, going into a convolution.
                nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # state size. 512 x 4 x 4
                nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. 256 x 8 x 8
                nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # state size. 128 x 16 x 16
                nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # state size. 64 x 32 x 32
                nn.ConvTranspose2d(64, 4, (4, 4), (2, 2), (1, 1), bias=True),
                nn.Tanh()
                # state size. 4 x 64 x 64
            )
        elif opt.gen_channel_size == 64:
            self.main = nn.Sequential(
                # Input is 100, going into a convolution.
                nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # state size. 512 x 4 x 4
                nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. 256 x 8 x 8
                nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # state size. 128 x 16 x 16
                nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # state size. 64 x 32 x 32
                nn.ConvTranspose2d(64, 8, (4, 4), (2, 2), (1, 1), bias=True),
                nn.Tanh()
                # state size. 8 x 64 x 64
            )
        elif opt.gen_channel_size == 48:
            self.main = nn.Sequential(
                # Input is 100, going into a convolution.
                nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # state size. 512 x 4 x 4
                nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. 256 x 8 x 8
                nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # state size. 128 x 16 x 16
                nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # state size. 64 x 32 x 32
                nn.ConvTranspose2d(64, 6, (4, 4), (2, 2), (1, 1), bias=True),
                nn.Tanh()
                # state size. 6 x 64 x 64
            )
        elif opt.gen_channel_size == 16:
            self.main = nn.Sequential(
                # Input is 100, going into a convolution.
                nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # state size. 512 x 4 x 4
                nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. 256 x 8 x 8
                nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # state size. 128 x 16 x 16
                nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # state size. 64 x 32 x 32
                nn.ConvTranspose2d(64, 2, (4, 4), (2, 2), (1, 1), bias=True),
                nn.Tanh()
                # state size. 2 x 64 x 64
            )
        elif opt.gen_channel_size == 8:
            self.main = nn.Sequential(
                # Input is 100, going into a convolution.
                nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # state size. 512 x 4 x 4
                nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. 256 x 8 x 8
                nn.Conv2d(256, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # state size. 128 x 8 x 8
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # state size. 64 x 8 x 8
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.Tanh()
                # state size. 256 x 8 x 8
            )
        elif opt.gen_channel_size == 9:
            self.main = nn.Sequential(
                # Input is 100, going into a convolution.
                nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # state size. 512 x 4 x 4
                nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. 256 x 8 x 8
                nn.Conv2d(256, 128, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                # state size. 128 x 8 x 8
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
                # state size. 64 x 8 x 8
                nn.Conv2d(64, 256, 3, stride=1, padding=1),
                nn.Tanh()
                # state size. 256 x 8 x 8
            )

        elif opt.gen_channel_size == 4:
            self.main = nn.Sequential(
                # Input is 100, going into a convolution.
                nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # state size. 512 x 4 x 4
                nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. 256 x 8 x 8
                nn.Conv2d(256, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # state size. 128 x 8 x 8
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # state size. 64 x 32 x 32
                nn.Conv2d(64, 8, 3, stride=1, padding=1),
                nn.Tanh()
                # state size. 8 x 8 x 8
            )
        elif opt.gen_channel_size == 2:
            self.main = nn.Sequential(
                # Input is 100, going into a convolution.
                nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # state size. 512 x 4 x 4
                nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. 256 x 8 x 8
                nn.Conv2d(256, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # state size. 128 x 8 x 8
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # state size. 64 x 32 x 32
                nn.Conv2d(64, 2, 3, stride=1, padding=1),
                nn.Tanh()
                # state size. 8 x 8 x 8
            )

    def forward(self, x):
        return self.main(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)