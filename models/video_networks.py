import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from subnet import *
import torchac

def save_model(model, iter):
    torch.save(model.state_dict(), "./snapshot/iter{}.model".format(iter))

def load_model(model, f):
    with open(f, 'rb') as f:
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

class VideoEncoder(nn.Module):
    def __init__(self):
        super(VideoEncoder, self).__init__()
        self.imageencoder = Image_Encoder()
        self.opticFlow = ME_Spynet()
        self.warpnet = Warp_net()

    def forwardFirstFrame(self, x):
        output, bittrans = self.imageCompressor(x)
        cost = self.bitEstimator(bittrans)
        return output, cost

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_image, referframe):
        estmv = self.opticFlow(input_image, referframe)
        prediction, warpframe = self.motioncompensation(referframe, estmv)
        input_residual = input_image - prediction

        fusion = torch.cat((estmv, prediction, input_residual), 2)
        encoded_feature = self.imageencoder(fusion)

        return encoded_feature

class VideoDecoder(nn.Module):
    def __init__(self):
        super(VideoDecoder, self).__init__()
        self.imagedecoder = Image_Decoder()
        self.opticFlow = ME_Spynet()
        self.warpnet = Warp_net()

    def forwardFirstFrame(self, x):
        output, bittrans = self.imageCompressor(x)
        cost = self.bitEstimator(bittrans)
        return output, cost

    def motioncompensation(self, ref, mv):
        warpframe = flow_warp(ref, mv)
        inputfeature = torch.cat((warpframe, ref), 1)
        prediction = self.warpnet(inputfeature) + warpframe
        return prediction, warpframe

    def forward(self, input_latent, referframe):
        estmv, input_residual = self.imagedecoder(input_latent)

        prediction, warpframe = self.motioncompensation(referframe, estmv)
        input_image = input_residual + prediction

        return input_image