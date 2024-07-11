import time
from models import create_seperate_model
import os
import numpy as np
import torch
import torchvision
from configs.config_attack import cfg
from mmcv import Config, DictAction
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from data.speech_dataset import *
from models.attack_basics import *
from torch.autograd import Variable
import pdb
import imageio
import random
import json
import math
from data.EurDataset import EurDataset, collate_data
from models.generator import Generator, Res_Generator
import logging
import shutil
import scipy.io.wavfile as wavfile
import warnings
warnings.filterwarnings("ignore")

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
norm_layer = DeNormalize(mean=[104, 117, 128], std=[1, 1, 1])
after_norm_layer = Normalize(mean=[104, 117, 128], std=[1, 1, 1])

cfg.is_attack = 'black'

vocab_file = cfg.vocab_file
vocab = json.load(open(vocab_file, 'rb'))
token_to_idx = vocab['token_to_idx']
num_vocab = len(token_to_idx)
pad_idx = token_to_idx["<PAD>"]
start_idx = token_to_idx["<START>"]
end_idx = token_to_idx["<END>"]
cfg.num_vocab = num_vocab
cfg.pad_idx = pad_idx
bleu_score_1gram = BleuScore(1, 0, 0, 0)
bleu_score_2gram = BleuScore(0, 1, 0, 0)
bleu_score_3gram = BleuScore(0, 0, 1, 0)
bleu_score_4gram = BleuScore(0, 0, 0, 1)


def validate(validate_dataset, dataset_name, morality, attack_type):
    netGen.eval()

    tx_sumpsnr = 0
    tx_summsssim = 0
    rx_sumpsnr = 0
    rx_summsssim = 0
    cnt = 0
    sum_MSE = 0
    score_1gram = []
    score_2gram = []
    score_3gram = []
    score_4gram = []
    Tx_word = []
    Rx_word = []
    word = []
    target_word = []

    if attack_type == 'random':
        cfg.is_attack = 'random'
    elif attack_type == 'black':
        cfg.is_attack = 'black'
    elif attack_type == 'no':
        cfg.is_attack = 'no'

    if morality == 'text':
        StoT = SeqtoText(token_to_idx, end_idx)
    
    for i, data in enumerate(validate_dataset):
        if i > 50:
            break

        if dataset_name == 'UCF':
            video = data['imgs'].to(device)
            video = norm_layer(video)
            div_video = torch.div(video, 255.0) # [1, 3, 16, 128, 128]
            video_len = div_video.size(2)

        elif dataset_name == 'Vimeo':
            input = data[0]
        else :
            input = data

        if morality == 'video':
            
            for frame_id in range(video_len):
                noise = torch.randn([cfg.gen_batch_size, 100, 1, 1], device=device)
                wireless_perturbation = netGen(noise)            
        
                input = div_video[:,:,frame_id,:,:]
                
                if frame_id % cfg.gop_len == 0:
                    size_latent = cfg.img_S
                    image_model.set_latent_size(size_latent)
                    image_model.set_input(input.repeat(cfg.how_many_channel, 1, 1, 1))
                    
                    if attack_type == 'black':
                        image_model.set_blackbox_perturbation(wireless_perturbation)
                    elif attack_type == 'random':
                        size_latent = int(np.ceil(cfg.batch_size * cfg.P * cfg.img_S * cfg.M / 48))
                        wireless_perturbation = torch.randn(cfg.gen_batch_size, cfg.P, size_latent+1, cfg.M + cfg.K, 2, dtype=torch.float32).to(device)
                        image_model.set_random_perturbation(wireless_perturbation)
                    else:
                        image_model.initialize_perturbation()
                    image_model.forward()
                    
                    fake = image_model.fake
                    tx_fake = image_model.tx_fake
                else:
                    size_latent = 2*(cfg.video_S)
                    video_model.set_latent_size(size_latent)
                    video_model.set_input(input, tx_referframe, rx_referframe)
                    
                    if attack_type == 'black':
                        video_model.set_blackbox_perturbation(wireless_perturbation)
                    elif attack_type == 'random':
                        size_latent = int(np.ceil(cfg.batch_size * cfg.P * cfg.video_S * 2 * cfg.M / 48))
                        wireless_perturbation = torch.randn(cfg.gen_batch_size, cfg.P, size_latent+1, cfg.M + cfg.K, 2, dtype=torch.float32).to(device)
                        video_model.set_random_perturbation(wireless_perturbation)
                    else:
                        video_model.initialize_perturbation()
                    video_model.forward()
                    
                    fake = video_model.rx_clipped_recon_image
                    tx_fake = video_model.tx_clipped_recon_image

                # [1, video_len, C, W, H]
                rx_PSNR, tx_PSNR, rx_ssim_val, tx_ssim_val = calculate_performance(cfg, fake, tx_fake, input)
                                                    
                tx_referframe = tx_fake.clone().detach()
                rx_referframe = fake.clone().detach()

                tx_sumpsnr += tx_PSNR
                tx_summsssim += tx_ssim_val
                rx_sumpsnr += rx_PSNR
                rx_summsssim += rx_ssim_val
                cnt += 1

        elif morality == 'image':
            for frame_id in range(video_len):
                input = div_video[:,:,frame_id,:,:]
                size_latent = cfg.img_S
                image_model.set_latent_size(size_latent)
                image_model.set_input(input.repeat(cfg.how_many_channel, 1, 1, 1))
                
                if attack_type == 'black':
                    noise = torch.randn([cfg.gen_batch_size, 100, 1, 1], device=device)
                    wireless_perturbation = netGen(noise)   
                    image_model.set_blackbox_perturbation(wireless_perturbation)
                elif attack_type == 'random':
                    size_latent = int(np.ceil(cfg.batch_size * cfg.P * cfg.img_S * cfg.M / 48))
                    wireless_perturbation = torch.randn(cfg.gen_batch_size, cfg.P, size_latent+1, cfg.M + cfg.K, 2, dtype=torch.float32).to(device)
                    image_model.set_random_perturbation(wireless_perturbation)
                else:
                    image_model.initialize_perturbation()
                image_model.forward()

                fake = image_model.fake
                tx_fake = image_model.tx_fake

                rx_PSNR, tx_PSNR, rx_ssim_val, tx_ssim_val = calculate_performance(cfg, fake, tx_fake, input)

                tx_sumpsnr += tx_PSNR
                tx_summsssim += tx_ssim_val
                rx_sumpsnr += rx_PSNR
                rx_summsssim += rx_ssim_val
                cnt += 1

        elif morality == 'text':
            cfg.text_S =  math.ceil(input.shape[1] * cfg.text_C_channel / 128)
            size_latent = cfg.text_S
            text_model.set_latent_size(size_latent)
            text_model.set_input(input)
            
            if attack_type == 'black':
                noise = torch.randn([cfg.gen_batch_size, 100, 1, 1], device=device)
                wireless_perturbation = netGen(noise)
                text_model.set_blackbox_perturbation(wireless_perturbation, cfg.text_S)
            elif attack_type == 'random':
                size_latent = int(np.ceil(cfg.batch_size * cfg.P * cfg.text_S * cfg.M / 48))
                wireless_perturbation = torch.randn(cfg.gen_batch_size, cfg.P, size_latent+1, cfg.M + cfg.K, 2, dtype=torch.float32).to(device)
                text_model.set_random_perturbation(wireless_perturbation)
            else:
                text_model.initialize_perturbation()
            out = text_model.greedy(start_idx)
            
            sentences = out.cpu().numpy().tolist()
            result_string = list(map(StoT.sequence_to_text, sentences))
            word = word + result_string
            
            target_sent = text_model.trg.cpu().numpy().tolist()
            result_string = list(map(StoT.sequence_to_text, target_sent))
            target_word = target_word + result_string

        elif morality == 'speech':
            input = input.view(cfg.gen_batch_size, -1)
            speech_model.set_input(input)
            speech_model.set_latent_size(cfg.speech_S)
            
            if attack_type == 'black':
                noise = torch.randn([cfg.gen_batch_size, 100, 1, 1], device=device)
                wireless_perturbation = netGen(noise)
                speech_model.set_blackbox_perturbation(wireless_perturbation)
            elif attack_type == 'random':
                size_latent = int(np.ceil(cfg.batch_size * cfg.P * cfg.speech_S * cfg.M / 48))
                wireless_perturbation = torch.randn(cfg.gen_batch_size, cfg.P, size_latent+1, cfg.M + cfg.K, 2, dtype=torch.float32).to(device)
                speech_model.set_random_perturbation(wireless_perturbation)
            else:
                speech_model.initialize_perturbation()
            speech_model.forward()
            
            sum_MSE += speech_model.MSE_speech
            cnt += 1

        torch.cuda.empty_cache()

    if (morality == 'image') | (morality == 'video'):
        log = " Attack: {}, Morality {}, RX psnr : {:.2f}, RX msssim: {:.3f}".format(attack_type, morality, rx_sumpsnr/cnt, rx_summsssim/cnt)
        print(log)
        return log
    elif morality == 'speech':
        log = " Attack: {}, Morality {}, RX MSE: {:.7f}".format(attack_type, morality, sum_MSE/cnt)
        print(log)     
        return log  
    elif morality == 'text':
        Tx_word.append(word)
        Rx_word.append(target_word)

        bleu_score_1g = []
        bleu_score_2g = []
        bleu_score_3g = []
        bleu_score_4g = []
        sim_score = []
        for sent1, sent2 in zip(Tx_word, Rx_word):
            # 1-gram
            bleu_score_1g.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) # 7*num_sent
            # 2-gram
            bleu_score_2g.append(bleu_score_2gram.compute_blue_score(sent1, sent2)) # 7*num_sent
            # 3-gram
            bleu_score_3g.append(bleu_score_3gram.compute_blue_score(sent1, sent2)) # 7*num_sent
            # 4-gram
            bleu_score_4g.append(bleu_score_4gram.compute_blue_score(sent1, sent2)) # 7*num_sent
        bleu_score_1g = np.array(bleu_score_1g)
        bleu_score_1g = np.mean(bleu_score_1g, axis=1)
        score_1gram.append(bleu_score_1g)
        score1 = np.mean(np.array(score_1gram), axis=0)

        bleu_score_2g = np.array(bleu_score_2g)
        bleu_score_2g = np.mean(bleu_score_2g, axis=1)
        score_2gram.append(bleu_score_2g)
        score2 = np.mean(np.array(score_2gram), axis=0)

        bleu_score_3g = np.array(bleu_score_3g)
        bleu_score_3g = np.mean(bleu_score_3g, axis=1)
        score_3gram.append(bleu_score_3g)
        score3 = np.mean(np.array(score_3gram), axis=0)

        bleu_score_4g = np.array(bleu_score_4g)
        bleu_score_4g = np.mean(bleu_score_4g, axis=1)
        score_4gram.append(bleu_score_4g)
        score4 = np.mean(np.array(score_4gram), axis=0)

        log1 = " Attack: {}, Morality {}, RX BLEU_1g: {} ".format(attack_type, morality, score1)
        log2 = " Attack: {}, Morality {}, RX BLEU_2g: {} ".format(attack_type, morality, score2)
        log3 = " Attack: {}, Morality {}, RX BLEU_3g: {} ".format(attack_type, morality, score3)
        log4 = " Attack: {}, Morality {}, RX BLEU_4g: {} ".format(attack_type, morality, score4)
        print(log1) 
        print(log2)
        print(log3)
        print(log4)

        return log1 + log2 + log3 + log4

def main():
    '''
    Load Dataset.
    '''

    # image & video
    ucf_cfg = Config.fromfile('configs/config_ucf.py')
    train_ucf_dataset = build_dataset(ucf_cfg.data.train, dict(test_mode=False,  # cfg.data.test: num clip=10 instead of 1
                                                sample_by_class=False))
    test_ucf_dataset = build_dataset(ucf_cfg.data.val, dict(test_mode=False,  # cfg.data.test: num clip=10 instead of 1
                                                sample_by_class=False))
    # text
    train_eur= EurDataset(cfg.text_data_dir, 'train')
    train_text_dataset = DataLoader(train_eur, batch_size=cfg.gen_batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    test_eur = EurDataset(cfg.text_data_dir, 'test')
    test_text_dataset = torch.utils.data.DataLoader(test_eur, batch_size=cfg.gen_batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)

    # speech
    train_dataset = Edinburgh_dataset(cfg.speech_data_dir, is_train=True)
    train_speech_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.gen_batch_size, shuffle=True, num_workers=0)
    test_dataset = Edinburgh_dataset(cfg.speech_data_dir, is_train=False)
    test_speech_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=cfg.gen_batch_size, num_workers=0)
    
    '''
    Load JSCC Model.
    '''
    global image_model, video_model, speech_model, text_model, netGen
    image_model, video_model, speech_model, text_model = \
        create_seperate_model(cfg)
    set_config(cfg)

    '''
    Load PGM
    '''
    if cfg.gen_network == 'DCGAN':
        netGen = Generator(cfg).to(device)
    else:
        netGen = Res_Generator().to(device)

    if cfg.gen_continuetrain == True:
        state_dict = torch.load(cfg.gen_dataroot, map_location=str(device))
        netGen.load_state_dict(state_dict)
    else:
        if cfg.gen_network == 'DCGAN':
            netGen._initialize_weights()

    global optimizer_netGen
    bp_parameters = netGen.parameters()
    optimizer_netGen = torch.optim.Adam(bp_parameters, cfg.attack_lr, (0.5, 0.999))

    text_folder_name = 'log' + f'_{cfg.adv_num_channel}' 
    if not os.path.isdir(text_folder_name):
        os.mkdir(text_folder_name)
    text_writer_name = text_folder_name + '/' + 'test_black_box_attack' + f'_{cfg.SNR}' + f'_{cfg.psr * -1}' + f'_{cfg.gen_channel_size}' + f'_{cfg.coding_rate}' + '_wo' + '.txt'
    
    # modulation
    mod = 1
    
    with open(text_writer_name, 'w', buffering=1) as f:        
        if (mod % 3) == 0:
            cfg.modulation = 'PSK'
            cfg.m_degree = 4
            image_model.channel.change_constellation()
            video_model.channel.change_constellation()
            speech_model.channel.change_constellation()
            text_model.channel.change_constellation()
            print('Modulation: QPSK')
            
        elif (mod % 3) == 1:
            cfg.modulation = 'QAM'
            cfg.m_degree = 16
            image_model.channel.change_constellation()
            video_model.channel.change_constellation()
            speech_model.channel.change_constellation()
            text_model.channel.change_constellation()
            print('Modulation: 16-QAM')
        elif (mod % 3) == 2:
            cfg.modulation = 'QAM'
            cfg.m_degree = 64
            image_model.channel.change_constellation()
            video_model.channel.change_constellation()
            speech_model.channel.change_constellation()
            text_model.channel.change_constellation()
            print('Modulation: 64-QAM')

        log = 'psr: ' + f'{cfg.psr}' + ' cd_rate: ' + f'{cfg.coding_rate}' + ' mod: ' + f'{cfg.modulation}' + ' degree: ' + f'{cfg.m_degree}'
        f.write(log + '\n')
        
        with torch.no_grad():

            log = validate(test_ucf_dataset, 'UCF', 'video', attack_type='random')
            f.write(log + '\n')
            log = validate(test_ucf_dataset, 'UCF', 'video', attack_type='black')
            f.write(log + '\n')

            log = validate(test_ucf_dataset, 'UCF', 'image', attack_type='random')
            f.write(log + '\n')
            log = validate(test_ucf_dataset, 'UCF', 'image', attack_type='black')
            f.write(log + '\n')

            log = validate(test_speech_dataset, 'Edinburgh', 'speech', attack_type='random')
            f.write(log + '\n')
            log = validate(test_speech_dataset, 'Edinburgh', 'speech', attack_type='black')
            f.write(log + '\n')

            log = validate(test_text_dataset, 'EurDataset', 'text', attack_type='random')
            f.write(log + '\n')
            log = validate(test_text_dataset, 'EurDataset', 'text', attack_type='black')
            f.write(log + '\n')
                
def set_config(cfg):

    if cfg.coding_rate == 1:
        cfg.img_C_channel                            = 64                       
        cfg.video_C_channel                          = 48                  
        cfg.speech_C_channel                         = 128           
        cfg.text_C_channel                           = 96
    else:
        cfg.img_C_channel                            = 96                       
        cfg.video_C_channel                          = 64                  
        cfg.speech_C_channel                         = 192           
        cfg.text_C_channel                           = 128    

    img_size_latent = (cfg.size_w // (2**3)) * (cfg.size_h // (2**3)) * (cfg.img_C_channel // 2)
    video_size_latent = (cfg.size_w // (2**4)) * (cfg.size_h // (2**4)) * (cfg.video_C_channel // 2)
    speech_size_latent = (cfg.size_w // (2**2)) * (cfg.size_h // (2**2)) * (cfg.speech_C_channel // 2)
    cfg.img_S                                        = img_size_latent // cfg.M          
    cfg.video_S                                      = video_size_latent // cfg.M 
    cfg.speech_S                                     = speech_size_latent // cfg.M  

    return cfg

if __name__ == "__main__":
    main()
