# Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.nn import functional as F
from torch.autograd import Function
from .resample import downsample2, upsample2
import pdb
import math
###############################################################################
# Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_net_text(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    for p in net.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return net

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)



class Normalize(nn.Module):
  def forward(self, x, power):
    N = x.shape[0]
    pwr = torch.mean(x**2, (1,2,3), True)

    return np.sqrt(power)*x/torch.sqrt(pwr)


# Initialization of encoder, generator and discriminator (optional)
def define_E(input_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Encoder(input_nc=input_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect")
    #net = VEncoder()
    return init_net(net, init_type, init_gain, gpu_ids)

def define_G(output_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm="instance", init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Generator(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect")
    #net = VDecoder()
    return init_net(net, init_type, init_gain, gpu_ids)

def define_Speech_E(opt, init_type='kaiming', norm="instance", init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    #net = Speech_Encoder(norm_layer=norm_layer)
    net = Speech_Encoder2(opt, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_Speech_G(opt, init_type='kaiming', norm="instance", init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    #net = Speech_Generator(norm_layer=norm_layer)
    net = Speech_Generator2(opt, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_Text_E(opt, num_vocab, gpu_ids=[]):
    net = None
    net = Text_JSCC_Encoder(opt, src_vocab_size=num_vocab, trg_vocab_size=num_vocab, src_max_len=num_vocab, trg_max_len=num_vocab)
    return init_net_text(net, gpu_ids)

def define_Text_G(opt, num_vocab, gpu_ids=[]):
    net = None
    net = Text_JSCC_Decoder(opt, src_vocab_size=num_vocab, trg_vocab_size=num_vocab, src_max_len=num_vocab, trg_max_len=num_vocab)
    return init_net_text(net, gpu_ids)

def define_TX_G(output_nc, ngf, max_ngf, n_downsample, C_channel, n_blocks, norm="instance", init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Generator(output_nc=output_nc, ngf=ngf, max_ngf=max_ngf, C_channel=C_channel, n_blocks=n_blocks, n_downsampling=n_downsample, norm_layer=norm_layer, padding_type="reflect")
    #net = VDecoder()
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(input_nc, ndf, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_S(dim, dim_out, dim_in=64, norm='instance', init_type='kaiming', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    net = Subnet(dim=dim, dim_out=dim_out, dim_in = dim_in, padding_type='zero', norm_layer=norm_layer, use_dropout=False)
    return init_net(net, init_type, init_gain, gpu_ids)

# Encoder network
class Encoder(nn.Module):

    def __init__(self, input_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        """Construct a Resnet-based encoder

        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the first conv layer
            max_ngf (int)       -- the maximum number of filters
            C_channel (int)     -- the number of channels of the output
            n_blocks (int)      -- the number of ResNet blocks
            n_downsampling      -- number of downsampling layers
            norm_layer          -- normalization layer
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_downsampling>=0)
        assert(n_blocks>=0)
        super(Encoder, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d((5-1)//2),
                 nn.Conv2d(input_nc, ngf, kernel_size=5, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 activation]

        # add downsampling layers
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(min(ngf * mult,max_ngf), min(ngf * mult * 2,max_ngf), kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(min(ngf * mult * 2, max_ngf)), activation]

        self.model_down = nn.Sequential(*model)
        model= []
        # add ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(min(ngf * mult,max_ngf), padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]

        self.model_res = nn.Sequential(*model)

        self.projection = nn.Conv2d(min(ngf * mult,max_ngf), C_channel, kernel_size=3, padding=1, stride=1, bias=use_bias)

    def forward(self, input, H=None):
        z = self.model_down(input)
        z = self.model_res(z) # [1, 256, 8, 8]
        y = self.projection(z) # [1, 12, 8, 8]
        return y


# Generator network
class Generator(nn.Module):
    def __init__(self, output_nc, ngf=64, max_ngf=512, C_channel=16, n_blocks=2, n_downsampling=2, norm_layer=nn.BatchNorm2d, padding_type="reflect"):
        """Construct a Resnet-based generator

        Parameters:
            output_nc (int)     -- the number of channels for the output image
            ngf (int)           -- the number of filters in the first conv layer
            max_ngf (int)       -- the maximum number of filters
            C_channel (int)     -- the number of channels of the input
            n_blocks (int)      -- the number of ResNet blocks
            n_downsampling      -- number of downsampling layers
            norm_layer          -- normalization layer
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks>=0)
        assert(n_downsampling>=0)

        super(Generator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        activation = nn.ReLU(True)

        mult = 2 ** n_downsampling
        ngf_dim = min(ngf * mult, max_ngf)
        model = [nn.Conv2d(C_channel,ngf_dim,kernel_size=3, padding=1 ,stride=1, bias=use_bias)]

        for i in range(n_blocks):
            model += [ResnetBlock(ngf_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=False, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(ngf * mult,max_ngf), min(ngf * mult //2, max_ngf),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(min(ngf * mult //2, max_ngf)),
                      activation]

        model += [nn.ReflectionPad2d((5-1)//2), nn.Conv2d(ngf, output_nc, kernel_size=5, padding=0)]

        model +=[nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):

        return 2*self.model(input)-1

'''
Speech
'''
class channel_change_block(nn.Module):
    def __init__(self, chin=1, hidden=64, kernel_size=9):
        super().__init__()
        self.conv1 = nn.Conv1d(chin, hidden, kernel_size, padding=4)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu1 = nn.PReLU(init=0.3)

        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, padding=4)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.relu2 = nn.PReLU(init=0.3)

        self.conv3 = nn.Conv1d(chin, hidden, kernel_size, padding=4)
        self.bn3 = nn.BatchNorm1d(hidden)
        self.relu3 = nn.PReLU(init=0.3)

        
    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.relu2(self.bn2(self.conv2(y)))
        z = self.relu3(self.bn3(self.conv3(x)))

        return y + z

class res_block(nn.Module):
    def __init__(self, dilation=1, hidden=64, kernel_size=9):
        super().__init__()
        padding = 4 * dilation
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size, padding=padding, dilation=dilation)
        #torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        #torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.relu1 = nn.PReLU(init=0.3)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, padding=padding, dilation=dilation)
        #torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        #torch.nn.init.constant_(self.conv2.bias.data, 0.0)
        self.relu2 = nn.PReLU(init=0.3)

    def forward(self, x):
        y = self.relu1(self.conv1(x))
        y = self.relu2(self.conv2(y))
        
        return x + y

class up_block(nn.Module):
    def __init__(self, hidden=64, kernel_size=9):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(hidden, hidden, kernel_size, stride=2, padding=4, output_padding=1)
        self.relu1 = nn.PReLU(init=0.3)
        self.bn1 = nn.BatchNorm1d(hidden)
        #torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        #torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, stride=1, padding=4)
        self.relu2 = nn.PReLU(init=0.3)
        self.bn2 = nn.BatchNorm1d(hidden)
        #torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        #torch.nn.init.constant_(self.conv2.bias.data, 0.0)
        self.conv3 = nn.ConvTranspose1d(hidden, hidden, kernel_size, stride=2, padding=4, output_padding=1)
        self.relu3 = nn.PReLU(init=0.3)
        self.bn3 = nn.BatchNorm1d(hidden)
        #torch.nn.init.xavier_uniform_(self.conv3.weight.data)
        #torch.nn.init.constant_(self.conv3.bias.data, 0.0)
        
    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.relu2(self.bn2(self.conv2(y)))
        z = self.relu3(self.bn3(self.conv3(x)))
        
        return y + z

class down_block(nn.Module):
    def __init__(self, hidden=64, kernel_size=9):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden, hidden, kernel_size, stride=2, padding=4)
        self.relu1 = nn.PReLU(init=0.3)
        self.bn1 = nn.BatchNorm1d(hidden)
        #torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        #torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size, stride=1, padding=4)
        self.relu2 = nn.PReLU(init=0.3)
        self.bn2 = nn.BatchNorm1d(hidden)
        #torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        #torch.nn.init.constant_(self.conv2.bias.data, 0.0)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size, stride=2, padding=4)
        self.relu3 = nn.PReLU(init=0.3)
        self.bn3 = nn.BatchNorm1d(hidden)
        #torch.nn.init.xavier_uniform_(self.conv3.weight.data)
        #torch.nn.init.constant_(self.conv3.bias.data, 0.0)
        
    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.relu2(self.bn2(self.conv2(y)))
        z = self.relu3(self.bn3(self.conv3(x)))
        
        return y + z


        
def Speech_Normalize(input):
    mean = torch.mean(input, dim=-1, keepdim=True)
    std = torch.std(input, dim=-1, keepdim=True)
    
    return ((input - mean) / std), mean, std

def Speech_Denormalize(input, mean, std):

    return ((input + mean) * std)

# Encoder network
class Speech_Encoder(nn.Module):

    def __init__(self,
                 chin=1,
                 depth=4,
                 norm_layer=nn.BatchNorm1d
                 ):

        super(Speech_Encoder, self).__init__()

        self.chin = chin
        self.depth = depth
        dils = [1, 2, 4, 8]
        
        model = [channel_change_block(chin=1, hidden=64)]
        
        '''
        for i in range(self.depth):
            dilation = dils[i]
            model += [ res_block(dilation=dilation) ]
        '''
        model += [ down_block() ]
        '''
        for i in range(self.depth):
            dilation = dils[i]
            model += [ res_block(dilation=dilation) ]
        '''
        model += [ channel_change_block(chin=64, hidden=1) ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x, H=None):
        x =  self.model(x)

        return x

# Encoder network
class Speech_Generator(nn.Module):

    def __init__(self,
                 chin=1,
                 depth=4,
                 norm_layer=nn.BatchNorm1d
                 ):

        super(Speech_Generator, self).__init__()

        self.chin = chin
        self.depth = depth
        dils = [1, 2, 4, 8]
        
        model = [channel_change_block(chin=1, hidden=64)]
        '''
        for i in range(self.depth):
            dilation = dils[i]
            model += [ res_block(dilation=dilation) ]
        '''
        model += [ up_block() ]
        '''
        for i in range(self.depth):
            dilation = dils[i]
            model += [ res_block(dilation=dilation) ]
        '''
        model += [ channel_change_block(chin=64, hidden=1) ]
        
        self.model = nn.Sequential(*model)


    def forward(self, x, H=None):
        x = self.model(x)

        return x


# Encoder network
class Speech_Encoder2(nn.Module):

    def __init__(self,
                 opt,
                 chin=1,
                 depth=4,
                 hidden=128,
                 norm_layer=nn.BatchNorm2d
                 ):

        super(Speech_Encoder2, self).__init__()

        self.chin = chin
        self.depth = depth
        
        # Semantic Encoder   
        model = [ nn.Conv2d(1, 32, kernel_size=5, padding=2, stride=1), 
                    nn.BatchNorm2d(32),
                    nn.ReLU(True) ]
        model += [ nn.Conv2d(32, hidden, kernel_size=5, padding=2, stride=1),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(True) ]
        for i in range(self.depth-2):
            model += [ nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, stride=2), 
                      nn.BatchNorm2d(hidden),
                      nn.ReLU(True) ]
        
        # Channel Encoder
        model += [nn.Conv2d(hidden, opt.speech_C_channel, kernel_size=5, padding=2, stride=1)]
        
        self.model = nn.Sequential(*model)

        indices_1 = np.tile(np.reshape(np.arange(0, 128), [1,128]), [128, 1])
        indices_2 = np.transpose(np.tile(np.reshape(np.arange(0, 128 * 128, 128), [1, 128]), [128, 1]))
        indices = np.add(indices_1, indices_2)
        self.index = np.reshape(indices, [1, 128 * 128])
        
    def forward(self, x, H=None):
        device = x.get_device()
        N = x.size(0)

        # step1
        x, mean, std = Speech_Normalize(x)
        indexs = np.tile(self.index, [N, 1])
        tensor_index = torch.tensor(indexs, dtype=torch.int64).to(device)
        frame_input = torch.gather(x, 1, tensor_index)
        # step2
        frame_input = torch.reshape(frame_input, [N, 128, 128])
        frame_input = torch.unsqueeze(frame_input, 1)

        x =  self.model(frame_input)
        #pdb.set_trace()

        return x, mean, std

# Encoder network
class Speech_Generator2(nn.Module):

    def __init__(self,
                 opt,
                 chin=128,
                 depth=4,
                 hidden=128,
                 norm_layer=nn.BatchNorm2d
                 ):

        super(Speech_Generator2, self).__init__()

        self.chin = chin
        self.depth = depth

        # Channel Decoder
        model = [nn.Conv2d(opt.speech_C_channel, hidden, kernel_size=5, padding=2, stride=1),
                 nn.BatchNorm2d(hidden),
                 nn.ReLU(True)]

        # Semantic Decoder
        for i in range(self.depth-2):
            model += [ nn.ConvTranspose2d(hidden, hidden, 5, stride=2, padding=2, output_padding=1),
                      #nn.BatchNorm2d(hidden),
                      nn.ReLU(True) ]

        model += [nn.Conv2d(hidden, hidden, kernel_size=5, padding=2, stride=1),
                 nn.BatchNorm2d(hidden),
                 nn.ReLU(True)]
        # Last layer
        model += [nn.Conv2d(hidden, 1, kernel_size=5, padding=2, stride=1)]

        self.model = nn.Sequential(*model)
        
    def forward(self, x, netE_mean, netE_std, H=None):
        N = x.size(0)
        x =  self.model(x)
        x = torch.unsqueeze(x, axis=1)

        #wav1 = torch.reshape(x[:, 0 : 128 - 1, 0 : 128], 
        #                [N, (128 - 1) * 128])
        
        #wav2 = torch.reshape(x[:, 128 - 1, 0 : 128], 
        #                [N, 128])
        #wav_output = torch.concat([wav1, wav2], axis = 1)
        #pdb.set_trace()
        x = x.view(N, -1)
        x = Speech_Denormalize(x,netE_mean, netE_std)
        return x

# Encoder network
class VEncoder(nn.Module):
    def __init__(self, C_channel=12):
        super(VEncoder, self).__init__()
        
        self.conv1 = ResnetBlock_s2(inputchannel=3, outputchannel=256)
        self.mod1 = modulation(256)
        self.conv2 = ResnetBlock_s1()

        self.conv3 = ResnetBlock_s2()
        self.mod2 = modulation(256)
        self.at1 = Simple_attention()

        self.conv4 = ResnetBlock_s1()

        self.conv5 = ResnetBlock_s2()
        self.mod3 = modulation(256)
        self.conv6 = ResnetBlock_s1()

        self.conv7 = ResnetBlock_s2(outputchannel=C_channel)
        self.mod4 = modulation(C_channel)
        self.at2 = Simple_attention(inputchannel=C_channel, outputchannel=C_channel)
        
    def forward(self, input, SNR=5):
        y1 = self.conv2(self.conv1(input))
        y2 = self.at1(self.conv3(y1))
        y3 = self.conv4(y2)
        y4 = self.conv6(self.conv5(y3))
        y5 = self.conv7(y4)
        y6 = self.at2(y5)

        return  y5

# Decoder network
class VDecoder(nn.Module):
    def __init__(self, C_channel=12):
        super(VDecoder, self).__init__()

        self.at1 = Simple_attention(inputchannel=C_channel, outputchannel=C_channel)
        self.conv1 = ResnetBlock_s1(inputchannel=C_channel, outputchannel=256)
        self.conv2 = ResnetBlock_up()
        self.mod1 = modulation(256)

        self.conv3 = ResnetBlock_s1()
        self.conv4 = ResnetBlock_up()
        self.mod2 = modulation(256)

        self.at2 = Simple_attention()
        self.conv5 = ResnetBlock_s1()
        self.conv6 = ResnetBlock_up()
        self.mod3 = modulation(256)

        self.conv7 = ResnetBlock_s1()
        self.conv8 = ResnetBlock_up(inputchannel=256, outputchannel=3)
        self.mod4 = modulation(256)

    def forward(self, input, SNR=5):
        y1 = self.at1(input)
        y2 = self.conv2(self.conv1(y1))
        y3 = self.conv4(self.conv3(y2))
        y4 = self.at2(y3)
        y5 = self.conv6(self.conv5(y4))
        y6 = self.conv8(self.conv7(y5))
        return y6

# Text

class Text_PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(Text_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model)) #math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x
  
class Text_MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(Text_MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        
        self.dense = nn.Linear(d_model, d_model)
        
        #self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)
        
        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)
        
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)
        
        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.num_heads * self.d_k)
             
        x = self.dense(x)
        x = self.dropout(x)
        
        return x
    
    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #print(mask.shape)
        if mask is not None:
            # 根据mask，指定位置填充 -1e9  
            scores += (mask * -1e9)
            # attention weights
        p_attn = F.softmax(scores, dim = -1)
        return torch.matmul(p_attn, value), p_attn
    
class Text_PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(Text_PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x
    
class Text_EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout = 0.1):
        super(Text_EncoderLayer, self).__init__()
        
        self.mha = Text_MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = Text_PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)
        
        return x
    
class Text_DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, d_model, num_heads, dff, dropout):
        super(Text_DecoderLayer, self).__init__()
        self.self_mha = Text_MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.src_mha = Text_MultiHeadedAttention(num_heads, d_model, dropout = 0.1)
        self.ffn = Text_PositionwiseFeedForward(d_model, dff, dropout = 0.1)
        
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        
        #self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        "Follow Figure 1 (right) for connections."
        #m = memory
        
        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)
        
        src_output = self.src_mha(x, memory, memory, trg_padding_mask) # q, k, v
        x = self.layernorm2(x + src_output)
        
        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x

    
class Text_Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, num_layers, src_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Text_Encoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = Text_PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([Text_EncoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
        
    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        # the input size of x is [batch_size, seq_len]
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)
        
        return x
        
class Text_Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len, 
                 d_model, num_heads, dff, dropout = 0.1):
        super(Text_Decoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = Text_PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([Text_DecoderLayer(d_model, num_heads, dff, dropout) 
                                            for _ in range(num_layers)])
    
    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
            
        return x

class Text_ChannelEncoder(nn.Module):
    def __init__(self, d_model, ch_in):
        super(Text_ChannelEncoder, self).__init__()
        self.linear1 = nn.Linear(d_model, 256)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256, ch_in)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.relu(x1)
        x3 = self.linear2(x2)
        
        return x3

class Text_ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(Text_ChannelDecoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)
        
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        
        output = self.layernorm(x1 + x5)

        return output

class Text_dense(nn.Module):
    def __init__(self, d_model, trg_vocab_size):
        super(Text_dense, self).__init__()
        self.linear1 = nn.Linear(d_model, trg_vocab_size)
        
    def forward(self, x):
        x1 = self.linear1(x)

        return x1

class Text_JSCC_Encoder(nn.Module):
    def __init__(self, opt, num_layers=4, src_vocab_size=1, trg_vocab_size=1, src_max_len=1, 
                 trg_max_len=1, d_model=128, num_heads=8, dff=512, dropout = 0.1):
        super(Text_JSCC_Encoder, self).__init__()
        self.encoder = Text_Encoder(num_layers, src_vocab_size, src_max_len, 
                               d_model, num_heads, dff, dropout)
        self.channel_encoder = Text_ChannelEncoder(d_model, opt.text_C_channel)
        
        if src_vocab_size == 1:
            raise Exception("Vocab_size is 1 in networks.py")
        
    def forward(self, x, x_mask):
        x1 = self.encoder(x, x_mask)
        x2 = self.channel_encoder(x1)

        return x2

class Text_JSCC_Decoder(nn.Module):
    def __init__(self, opt, num_layers=4, src_vocab_size=1, trg_vocab_size=1, src_max_len=1, 
                 trg_max_len=1, d_model=128, num_heads=8, dff=512, dropout = 0.1):
        super(Text_JSCC_Decoder, self).__init__()
        self.channel_decoder = Text_ChannelDecoder(opt.text_C_channel, d_model, 512)
        self.decoder = Text_Decoder(num_layers, trg_vocab_size, trg_max_len, 
                               d_model, num_heads, dff, dropout)
        self.dense = Text_dense(d_model, trg_vocab_size)
        
        if src_vocab_size == 1:
            raise Exception("Vocab_size is 1 in networks.py")

    def forward(self, x, trg_inp, look_ahead_mask, src_mask):
        x1 = self.channel_decoder(x)
        x2 = self.decoder(trg_inp, x1, look_ahead_mask, src_mask)
        x3 = self.dense(x2)

        return x3

# Defines the resnet block
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

# Residual block stride 1 for Video JSCC
class ResnetBlock_s1(nn.Module):
    def __init__(self, inputchannel=256, outputchannel=256, kernel_size=3):
        super(ResnetBlock_s1, self).__init__()
        self.fea_ext = nn.Conv2d(inputchannel, outputchannel, 1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(inputchannel, outputchannel, kernel_size, 1, padding=kernel_size//2)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel, kernel_size, 1, padding=kernel_size//2)
        self.relu2 = nn.LeakyReLU()

    def forward(self, x):
        firstlayer = self.relu1(self.conv1(x))
        seclayer = self.conv2(firstlayer)
        return self.relu2(self.fea_ext(x) + seclayer)

# Residual block stride 2
class ResnetBlock_s2(nn.Module):
    def __init__(self, inputchannel=256, outputchannel=256, kernel_size=3):
        super(ResnetBlock_s2, self).__init__()
        self.fea_ext = nn.Conv2d(inputchannel, outputchannel, 1, stride=2, padding=0)
        self.conv1 = nn.Conv2d(inputchannel, outputchannel, kernel_size, 2, padding=kernel_size//2)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel, kernel_size, 1, padding=kernel_size//2)
        self.gdn2 = GDN(outputchannel)

    def forward(self, x):
        firstlayer = self.relu1(self.conv1(x))
        seclayer = self.conv2(firstlayer)
        return self.gdn2(self.fea_ext(x) + seclayer)

# Residual block stride 2
class ResnetBlock_up(nn.Module):
    def __init__(self, inputchannel=256, outputchannel=256, kernel_size=3):
        super(ResnetBlock_up, self).__init__()
        self.conv1_1 = nn.Conv2d(inputchannel, 4*outputchannel, kernel_size, 1, padding=kernel_size//2)
        self.conv1_2 = nn.Conv2d(inputchannel, 4*outputchannel, kernel_size, 1, padding=kernel_size//2)
        self.conv2_1 = nn.PixelShuffle(upscale_factor=2)
        self.conv2_2 = nn.PixelShuffle(upscale_factor=2)
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(outputchannel, outputchannel, kernel_size, 1, padding=kernel_size//2)
        self.gdn3 = GDN(outputchannel)

    def forward(self, x):
        y1 = self.conv1_1(x)
        y2 = self.conv2_1(y1)
        y3 = self.conv1_2(x)
        y4 = self.relu2(self.conv2_2(y3))
        y5 = self.conv3(y4)
        return self.gdn3(y2 + y5)

# Simple attention
class Simple_attention(nn.Module):
    def __init__(self, inputchannel=256, outputchannel=256):
        super(Simple_attention, self).__init__()
        self.trunk = nn.Sequential(ResnetBlock_s1(inputchannel, outputchannel), ResnetBlock_s1(inputchannel, outputchannel),
                        ResnetBlock_s1(inputchannel, outputchannel))

        self.mask = nn.Sequential(ResnetBlock_s1(inputchannel, outputchannel), ResnetBlock_s1(inputchannel, outputchannel),
                        ResnetBlock_s1(inputchannel, outputchannel), 
                        nn.Conv2d(inputchannel, outputchannel, 1, 1, padding=0))
    def forward(self, x):
        x1 = self.trunk(x)
        x2 = self.mask(x)
        x3 = (F.sigmoid(x2) * x1) + x
        return x3

class modulation(nn.Module):
    def __init__(self, C_channel):

        super(modulation, self).__init__()

        activation = nn.ReLU(True)

        # Policy network
        model_multi = [nn.Linear(C_channel + 1, C_channel), activation,
                       nn.Linear(C_channel, C_channel), nn.Sigmoid()]

        model_add = [nn.Linear(C_channel + 1, C_channel), activation,
                     nn.Linear(C_channel, C_channel)]

        self.model_multi = nn.Sequential(*model_multi)
        self.model_add = nn.Sequential(*model_add)

    def forward(self, z, SNR):

        # Policy/gate network
        N, C, W, H = z.shape

        z_mean = torch.mean(z, (-2, -1))
        z_cat = torch.cat((z_mean, SNR), -1)

        factor = self.model_multi(z_cat).view(N, C, 1, 1)
        addition = self.model_add(z_cat).view(N, C, 1, 1)

        return z * factor + addition

# Discriminator network
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        self.n_layers = n_layers

        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]]  # output 1 channel prediction map


        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))


    def forward(self, input):
        """Standard forward."""
        res = [input]
        for n in range(self.n_layers+1):
            model = getattr(self, 'model'+str(n))
            res.append(model(res[-1]))

        model = getattr(self, 'model'+str(self.n_layers+1))
        out = model(res[-1])

        return res[1:], out


# Different types of adversarial losses
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'none']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label.to(torch.float32)
        else:
            target_tensor = self.fake_label.to(torch.float32)
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# Subnets 
class Subnet(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, dim_out, dim_in, padding_type, norm_layer, use_dropout):

        super(Subnet, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.conv_block = self.build_conv_block(dim, dim_out, dim_in, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, dim_out, dim_in, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim_in, kernel_size=5, padding=2, bias=use_bias), norm_layer(dim_in), nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim_in, dim_in, kernel_size=5, padding=2, bias=use_bias), norm_layer(dim_in), nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim_in, dim_out, kernel_size=5, padding=2, bias=use_bias)]
        return nn.Sequential(*conv_block)

    def forward(self, x): 
        return self.conv_block(x) 



class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
        
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None

class GDN(nn.Module):
    """Generalized divisive normalization layer.
    y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """

    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1,
                 reparam_offset=2**-18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = ((self.beta_min + self.reparam_offset**2)**0.5)
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch)
        g = self.gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs

class soft_to_hard_quantize(nn.Module):
    def __init__(self, hardness):
        super(soft_to_hard_quantize, self).__init__()
        self.softmax = nn.Softmax(dim=0)
        self.hardness = hardness
        
    def forward(self, input, ref):
        #input : [batch, pilot, Number of packets, Number of subcarriers per symbol, 2]
        #ref : [M-, 2]
        
        M = ref.size(0)
        D1, D2, D3, D4, D5 = input.size()

        distance = torch.zeros((M, D1, D2, D3, D4, D5)).to(input.device)
        z = torch.zeros_like(input)
        
        for j in range(M):
            distance[j] = input - ref[j,:]
        square_l2_norm = torch.sum(distance**2, -1)
        y = self.softmax(square_l2_norm * self.hardness * -1)
        y = torch.unsqueeze(y,5)

        for j in range(M):
            z += torch.mul(y[j], ref[j])

        return z

class mrc(nn.Module):
    def __init__(self, hardness):
        super(mrc, self).__init__()
        self.softmax = nn.Softmax(dim=0)
        self.hardness = hardness
        
    def forward(self, input, ref):
        #input : [batch, pilot, Number of packets, Number of subcarriers per symbol, 2]
        #ref : [M-, 2]

        M = ref.size(0)
        D1, D2, D3, D4, D5 = input.size()

        distance = torch.zeros((M, D1, D2, D3, D4, D5)).to(input.device)
        z = torch.zeros_like(input)
        
        for j in range(M):
            distance[j] = torch.abs(input - ref[j,:])
        rank = torch.argmin(a, dim=0)

        y = self.softmax(square_l2_norm * self.hardness * -1)
        y = torch.unsqueeze(y,5)

        for j in range(M):
            z += torch.mul(y[j], ref[j])

        return z