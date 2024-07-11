import time
from models import create_seperate_model
import util.util as util
import os
import numpy as np
import torch
import sys
from configs.config_attack import cfg
from data.speech_dataset import *
from models.attack_basics import *
import pdb
import random
import json
import math
import logging
import argparse
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

parser = argparse.ArgumentParser(description='DVC reimplement')

def restore_modulated_singal(attack_type, morality, mod):

    if attack_type == 'black':
        cfg.is_attack = 'black'
        STORE_FOLDER = 'SDR_results/black/'
    else:
        cfg.is_attack = 'no'
        STORE_FOLDER = 'SDR_results/no/'

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

    if morality == 'text':
        StoT = SeqtoText(token_to_idx, end_idx)
    
    video_len = 16

    if morality == 'video':
        
        for frame_id in range(video_len):         

            if mod == 0:
                ori_path = STORE_FOLDER + morality + '/' + 'ori_' + 'PSK' + '_' + "{}".format(frame_id) + '.txt'
                wireless_path = STORE_FOLDER + morality + '/' + 'wir_' + 'PSK' + '_' + "{}".format(frame_id) + '.txt'
                ref_path = STORE_FOLDER + morality + '/' + 'ref_' + 'PSK' + '_' + "{}".format(frame_id) + '.txt'
            elif mod == 1:
                ori_path = STORE_FOLDER + morality + '/' + 'ori_' + '16QAM' + '_' + "{}".format(frame_id) + '.txt'
                wireless_path = STORE_FOLDER + morality + '/' + 'wir_' + '16QAM' + '_' + "{}".format(frame_id) + '.txt'
                ref_path = STORE_FOLDER + morality + '/' + 'ref_' + '16QAM' + '_' + "{}".format(frame_id) + '.txt'
            else:
                ori_path = STORE_FOLDER + morality + '/' + 'ori_' + '64QAM' + '_' + "{}".format(frame_id) + '.txt'
                wireless_path = STORE_FOLDER + morality + '/' + 'wir_' + '64QAM' + '_' + "{}".format(frame_id) + '.txt'
                ref_path = STORE_FOLDER + morality + '/' + 'ref_'  + '64QAM' + '_' + "{}".format(frame_id) + '.txt'
            
            input = torch.from_numpy(np.loadtxt(ori_path).reshape((1, 3, 128, 128))).type('torch.FloatTensor')
        
            if frame_id % cfg.gop_len == 0:
                size_latent = cfg.img_S
                image_model.set_latent_size(size_latent)
                image_model.set_input(input.repeat(cfg.how_many_channel, 1, 1, 1))
                image_model.channel.print_modulated_signal(wireless_path, ref_path)

                image_model.SDR_forward()
                
                fake = image_model.fake
                tx_fake = image_model.tx_fake
            else:
                size_latent = 2*(cfg.video_S)
                video_model.set_latent_size(size_latent)
                video_model.set_input(input, tx_referframe, rx_referframe)
                video_model.channel.print_modulated_signal(wireless_path, ref_path)
                
                video_model.SDR_forward()
                
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

            if mod == 0:
                ori_path = STORE_FOLDER + morality + '/' + 'ori_' + 'PSK' + '_' + "{}".format(frame_id) + '.txt'
                wireless_path = STORE_FOLDER + morality + '/' + 'wir_' + 'PSK' + '_' + "{}".format(frame_id) + '.txt'
                ref_path = STORE_FOLDER + morality + '/' + 'ref_' + 'PSK' + '_' + "{}".format(frame_id) + '.txt'
            elif mod == 1:
                ori_path = STORE_FOLDER + morality + '/' + 'ori_' + '16QAM' + '_' + "{}".format(frame_id) + '.txt'
                wireless_path = STORE_FOLDER + morality + '/' + 'wir_' + '16QAM' + '_' + "{}".format(frame_id) + '.txt'
                ref_path = STORE_FOLDER + morality + '/' + 'ref_' + '16QAM' + '_' + "{}".format(frame_id) + '.txt'
            else:
                ori_path = STORE_FOLDER + morality + '/' + 'ori_' + '64QAM' + '_' + "{}".format(frame_id) + '.txt'
                wireless_path = STORE_FOLDER + morality + '/' + 'wir_' + '64QAM' + '_' + "{}".format(frame_id) + '.txt'
                ref_path = STORE_FOLDER + morality + '/' + 'ref_'  + '64QAM' + '_' + "{}".format(frame_id) + '.txt'
            
            input = torch.from_numpy(np.loadtxt(ori_path).reshape((1, 3, 128, 128))).type('torch.FloatTensor')

            size_latent = cfg.img_S
            image_model.set_latent_size(size_latent)
            image_model.channel.print_modulated_signal(wireless_path, ref_path)
            
            image_model.SDR_forward()
            
            fake = image_model.fake
            tx_fake = image_model.tx_fake

            rx_PSNR, tx_PSNR, rx_ssim_val, tx_ssim_val = calculate_performance(cfg, fake, tx_fake, input)

            tx_sumpsnr += tx_PSNR
            tx_summsssim += tx_ssim_val
            rx_sumpsnr += rx_PSNR
            rx_summsssim += rx_ssim_val
            cnt += 1
            
    elif morality == 'text':
        size_latent =  math.ceil(input.shape[1] * cfg.text_C_channel / 128)
        text_model.set_latent_size(size_latent)
        text_model.set_input(input)
                    
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
        
        speech_model.forward()
        
        sum_MSE += speech_model.MSE_speech
        cnt += 1

    torch.cuda.empty_cache()

    if (morality == 'image') | (morality == 'video'):
        log = "Validation, Attack: {}, Morality: {}, TX psnr : {:.2f}, TX msssim: {:.3f}, RX psnr : {:.2f}, RX msssim: {:.3f}".format(attack_type, morality, tx_sumpsnr/cnt, tx_summsssim/cnt, rx_sumpsnr/cnt, rx_summsssim/cnt)
        print(log)
    elif morality == 'speech':
        log = "Validation, Attack: {}, Morality {}, MSE: {:.7f}".format(attack_type, morality, sum_MSE/cnt)
        print(log)   
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
            bleu_score_1g.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) 
            # 2-gram
            bleu_score_2g.append(bleu_score_2gram.compute_blue_score(sent1, sent2)) 
            # 3-gram
            bleu_score_3g.append(bleu_score_3gram.compute_blue_score(sent1, sent2)) 
            # 4-gram
            bleu_score_4g.append(bleu_score_4gram.compute_blue_score(sent1, sent2))

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

        log1 = "Validation, Attack: {}, Morality {}, BLEU: {}".format(attack_type, morality, score1)
        log2 = "Validation, Attack: {}, Morality {}, BLEU: {}".format(attack_type, morality, score2)
        log3 = "Validation, Attack: {}, Morality {}, BLEU: {}".format(attack_type, morality, score3)
        log4 = "Validation, Attack: {}, Morality {}, BLEU: {}".format(attack_type, morality, score4)
        print(log1) 
        print(log2)
        print(log3)
        print(log4)

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
 
def main():

    # Load Multimodal JSCC Models
    global image_model, video_model, speech_model, text_model, netGen
    image_model, video_model, speech_model, text_model = \
        create_seperate_model(cfg)
    set_config(cfg)
    
    args = parser.parse_args()
    
    with torch.no_grad():
        mod = 1

        if (mod % 3) == 0:
            cfg.modulation = 'PSK'
            cfg.m_degree = 4
            image_model.channel.change_constellation()
            video_model.channel.change_constellation()
            speech_model.channel.change_constellation()
            text_model.channel.change_constellation()
            
        elif (mod % 3) == 1:
            cfg.modulation = 'QAM'
            cfg.m_degree = 16
            image_model.channel.change_constellation()
            video_model.channel.change_constellation()
            speech_model.channel.change_constellation()
            text_model.channel.change_constellation()
        elif (mod % 3) == 2:
            cfg.modulation = 'QAM'
            cfg.m_degree = 64
            image_model.channel.change_constellation()
            video_model.channel.change_constellation()
            speech_model.channel.change_constellation()
            text_model.channel.change_constellation()

        restore_modulated_singal('no', 'image', mod)
        restore_modulated_singal('black', 'image', mod)

if __name__ == "__main__":
    main()
