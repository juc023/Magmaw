from easydict import EasyDict as edict
from configs.config import cfg
import os
import shutil

__E                                              = cfg

############################# Attack settings ###################################
__E.is_attack                                    = 'black' # ['no', 'random', 'black']
__E.iter_attack                                  = 200
__E.attack_lr                                    = 0.0002         
__E.psr                                          = -16
__E.gop_len                                      = 8
__E.gen_continuetrain                            = True
__E.adv_num_channel                              = 60
__E.gen_batch_size                               = 1
__E.gen_channel_size                             = 8 # [4, 16, 32 (basic), 64]
__E.gen_network                                  = 'DCGAN' # [Resnet, DCGAN]
__E.gen_time_offset                              = 2

# Model config for different datasets
__E.gen_dataroot                                 = __E.HOME_PATH + '/' + __E.gen_network + '/Generator_wo/adv' + f'_{__E.psr * -1}' + f'_{__E.gen_channel_size}' + f'_{__E.coding_rate}' + '.pth'

if __E.dataset_mode == 'Text':
     __E.data_type                               = 'text'

if __E.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __E.batch_size                               = 1
    __E.size_w                                   = 32
    __E.size_h                                   = 32 

elif __E.dataset_mode == 'CelebA':
    __E.batch_size                               = 1
    __E.dataroot                                 = './data/celeba/CelebA_test'
    __E.size_w                                   = 64
    __E.size_h                                   = 64  

elif __E.dataset_mode == 'OpenImage':
    __E.batch_size                               = 1
    __E.dataroot                                 = './data/Kodak'
    __E.size_w                                   = 512
    __E.size_h                                   = 768     

elif __E.dataset_mode == 'Vimeo':                           
    __E.batch_size                               = 1
    __E.dataroot                                 = ''
    __E.size_w                                   = 256
    __E.size_h                                   = 256   

elif (__E.dataset_mode == 'UCF') | (__E.dataset_mode == 'Multimodal'):                           
    __E.batch_size                               = 1
    __E.dataroot                                 = ''
    __E.size_w                                   = 128
    __E.size_h                                   = 128  

elif __E.dataset_mode == 'Speech':
    __E.batch_size                               = 1
    __E.dataroot                                 = ''
    __E.size_w                                   = 128
    __E.size_h                                   = 128   

elif __E.dataset_mode == 'Text':
    __E.batch_size                               = 1

__E.verbose                                      = False
__E.serial_batches                               = True
__E.isTrain                                      = False     

__E.num_test                                     = 50         # Number of images to test
__E.how_many_channel                             = 1           # Number of channel realizations per image
__E.epoch                                        = 'latest'    # Each model to use for testing
__E.load_iter                                    = 0


############################# OFDM configs ####################################
__E.P                                            = 1                                   # Number of packet
__E.M                                            = 64                                  # Number of subcarriers per symbol
__E.K                                            = 16                                  # Length of CP
__E.L                                            = 8                                   # Number of paths
__E.decay                                        = 4                                   # Exponential decay for the multipath channel


img_size_latent = (__E.size_w // (2**3)) * (__E.size_h // (2**3)) * (__E.img_C_channel // 2)
video_size_latent = (__E.size_w // (2**4)) * (__E.size_h // (2**4)) * (__E.video_C_channel // 2)
speech_size_latent = (__E.size_w // (2**2)) * (__E.size_h // (2**2)) * (__E.speech_C_channel // 2)
__E.img_S                                        = img_size_latent // __E.M            # Number of symbols
__E.video_S                                      = video_size_latent // __E.M          # Number of symbols
__E.speech_S                                     = speech_size_latent // __E.M         # Number of symbols 