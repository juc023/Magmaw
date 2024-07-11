from easydict import EasyDict as edict
import os
__C                                              = edict()
cfg                                              = __C

############################# Basic settings ####################################
HOME_DIR = os.path.expanduser("~") + '/'
USER_NAME = 'jungwoo'

__C.name                                         = 'JSCC_OFDM'      # Name of the experiment
__C.gpu_ids                                      = [0]              # GPUs to use
__C.dataset_mode                                 = 'Multimodal' 
__C.HOME_PATH                                    = HOME_DIR + 'magmaw/Checkpoints/Multimodal-JSCC'
__C.checkpoints_dir                              = __C.HOME_PATH + '/Vimeo'
__C.image_checkpoints_dir                        = __C.HOME_PATH + '/Vimeo'          
__C.video_checkpoints_dir                        = __C.HOME_PATH + '/Video_Vimeo'     
__C.text_checkpoints_dir                         = __C.HOME_PATH + '/Text'   
__C.speech_checkpoints_dir                       = __C.HOME_PATH + '/Speech' 
__C.flow_pretrain_np_dir                         = __C.HOME_PATH + '/flow_pretrain_np/'

__C.HOME_DATA_PATH                               = '/scr/' + USER_NAME + '/magmaw'
__C.image_data_dir                               = __C.HOME_DATA_PATH + '/ucf101/'
__C.video_data_dir                               = __C.HOME_DATA_PATH + '/ucf101/'
__C.text_data_dir                                = __C.HOME_DATA_PATH + '/text_data/'
__C.speech_data_dir                              = __C.HOME_DATA_PATH + '/speech_data/'

# Downstream
__C.setting_dir                                  = HOME_DIR + 'magmaw/Checkpoints/Multimodal-JSCC/Downstream/'

__C.model                                        = 'JSCCOFDM'
__C.coding_rate                                  = 1

if __C.coding_rate == 1:
    __C.img_C_channel                                = 64          
    __C.video_C_channel                              = 48   
    __C.speech_C_channel                             = 128
    __C.text_C_channel                               = 96 
else:
    __C.img_C_channel                                = 96            
    __C.video_C_channel                              = 64       
    __C.speech_C_channel                             = 192  
    __C.text_C_channel                               = 128  
    
# train                        
__C.SNR                                          = 10         # Signal to noise ratio

__C.img_SNR                                      = 10
__C.video_SNR                                    = 10
__C.text_SNR                                     = 10
__C.speech_SNR                                   = 10
# test
__C.test_SNR                                     = 10

__C.SNR_cal                                      = 'ins'      # ['ins', 'avg']. 'ins' is for instantaneous SNR, 'avg' is for average SNR
__C.feedforward                                  = 'OFDM-CE-EQ'  # Different schemes: 
                                                                         # OFDM-CE-EQ: MMSE channel estimation and equalization without any subnets
                                                                         # OFDM-CE-sub-EQ: MMSE channel estimation and equalization with CE subnet
                                                                         # OFDM-CE-sub-EQ-sub: MMSE channel estimation and equalization with CE & EQ subnet
                                                                         # OFDM-feedback: pre-coding scheme with CSI feedback
__C.N_pilot                                      = 1          # Number of pilot symbols
__C.is_clip                                      = False      # Whether to apply signal clipping or not
__C.CR                                           = 1.2        # Clipping ratio if clipping is applied
__C.lam_h                                        = 50         # Weight for the channel reconstruction loss
__C.gan_mode                                     = 'none'     # ['wgangp', 'lsgan', 'vanilla', 'none']
__C.lam_G                                        = 0.02       # Weight for the adversarial loss
__C.lam_L2                                       = 100        # Weight for image reconstruction loss
__C.lam_L1                                       = 1000        # Weight for image reconstruction loss

__C.modulation                                   = 'QAM' #[QPSK, QAM]
__C.m_degree                                     = 16
__C.write_constell_to_text                       = False

# Figures
__C.print_constell                               = False
__C.draw_power_spectrum                          = False
__C.draw_time_domain                             = False
__C.print_histogram                              = False

if __C.modulation == 'False':
    __C.checkpoints_dir                             += f'/No_mod'
    __C.image_checkpoints_dir                       += f'/No_mod'
    __C.video_checkpoints_dir                       += f'/No_mod'
    __C.text_checkpoints_dir                        += f'/No_mod'
    __C.speech_checkpoints_dir                      += f'/No_mod'
else:
    __C.checkpoints_dir                             += f'/{__C.m_degree}_{__C.modulation}'
    __C.image_checkpoints_dir                       += f'/{__C.m_degree}_{__C.modulation}'
    __C.video_checkpoints_dir                       += f'/{__C.m_degree}_{__C.modulation}'
    __C.text_checkpoints_dir                        += f'/{__C.m_degree}_{__C.modulation}'
    __C.speech_checkpoints_dir                      += f'/{__C.m_degree}_{__C.modulation}'

############################# Model and training configs ####################################

if __C.dataset_mode in ['CIFAR10', 'CIFAR100']:
    __C.n_layers_D                               = 3          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 2          # Downsample times
    __C.n_blocks                                 = 2          # Numebr of residual blocks


elif __C.dataset_mode == 'CelebA':
    __C.n_layers_D                               = 3          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 3          # Downsample times
    __C.n_blocks                                 = 2          # Numebr of residual blocks


elif __C.dataset_mode == 'OpenImage':
    __C.n_layers_D                               = 4          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 3          # Downsample times
    __C.n_blocks                                 = 4          # Numebr of residual blocks

elif __C.dataset_mode == 'Vimeo':
    if __C.data_type == 'image':   
        __C.n_layers_D                           = 4          # Number of layers in the discriminator. Only used with GAN loss
        __C.n_downsample                         = 3          # Downsample times
        __C.n_blocks                             = 4          # Numebr of residual blocks
    else:
        __C.n_layers_D                           = 4          # Number of layers in the discriminator. Only used with GAN loss
        __C.n_downsample                         = 4          # Downsample times
        __C.n_blocks                             = 3          # Numebr of residual blocks

elif (__C.dataset_mode == 'UCF') | (__C.dataset_mode == 'Multimodal') :    
    __C.n_layers_D                               = 4          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 3          # Downsample times
    __C.n_blocks                                 = 4          # Numebr of residual blocks
    __C.n_video_layers_D                         = 4          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_video_downsample                       = 4          # Downsample times
    __C.n_video_blocks                           = 3          # Numebr of residual blocks

elif __C.dataset_mode == 'Video_Vimeo':    
    __C.n_layers_D                               = 4          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 4          # Downsample times
    __C.n_blocks                                 = 3          # Numebr of residual blocks

elif __C.dataset_mode == 'Speech':
    __C.n_layers_D                               = 4          # Number of layers in the discriminator. Only used with GAN loss
    __C.n_downsample                             = 2          # Downsample times
    __C.n_blocks                                 = 4          # Numebr of residual blocks

__C.norm_D                                       = 'instance' if __C.gan_mode == 'wgangp' else 'batch'   # Type of normalization in Discriminator
__C.norm_EG                                      = 'batch'        # Type of normalization in Encoder and Generator



############################# Display and saving configs ####################################

__C.img_name = f'C{__C.img_C_channel}_{__C.feedforward}_SNR_{__C.img_SNR}_{__C.SNR_cal}_pilot_{__C.N_pilot}_hloss_{__C.lam_h}'
__C.img_name += f'_clip_{__C.CR}' if __C.is_clip else ''
__C.img_name += f'_{__C.gan_mode}_{__C.lam_G}' if __C.gan_mode != 'none' else ''

__C.video_name = f'C{__C.video_C_channel}_{__C.feedforward}_SNR_{__C.video_SNR}_{__C.SNR_cal}_pilot_{__C.N_pilot}_hloss_{__C.lam_h}'
__C.video_name += f'_clip_{__C.CR}' if __C.is_clip else ''
__C.video_name += f'_{__C.gan_mode}_{__C.lam_G}' if __C.gan_mode != 'none' else ''

__C.speech_name = f'C{__C.speech_C_channel}_{__C.feedforward}_SNR_{__C.speech_SNR}_{__C.SNR_cal}_pilot_{__C.N_pilot}_hloss_{__C.lam_h}'
__C.speech_name += f'_clip_{__C.CR}' if __C.is_clip else ''
__C.speech_name += f'_{__C.gan_mode}_{__C.lam_G}' if __C.gan_mode != 'none' else ''

__C.text_name = f'C{__C.text_C_channel}_{__C.feedforward}_SNR_{__C.text_SNR}_{__C.SNR_cal}_pilot_{__C.N_pilot}_hloss_{__C.lam_h}'
__C.text_name += f'_clip_{__C.CR}' if __C.is_clip else ''
__C.text_name += f'_{__C.gan_mode}_{__C.lam_G}' if __C.gan_mode != 'none' else ''


__C.vocab_file = __C.text_data_dir + 'vocab.json'
__C.MUSIC_test_csv_path = __C.HOME_DATA_PATH + '/AV-Robustness-CVPR21/data/test_AVE.csv'
__C.MUSIC_model_path = __C.HOME_DATA_PATH + '/AV-Robustness-CVPR21/ckpt'
__C.MUSIC_model_folder_name = 'AVE_av_reg-resnet18-anet-concat-learn-frames1stride8-audio16384rate8000-maxpool-epoch30-step10_20'
__C.MUSIC_audio_path = __C.HOME_DATA_PATH + '/AVE_Dataset/data/8k_audio'
__C.MUSIC_frame_path = __C.HOME_DATA_PATH + '/AVE_Dataset/data/frames'
