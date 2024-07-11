import torch
import numpy as np
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import imageio
import pdb
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
import torchvision.transforms.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx
        
    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return(words) 
    
class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1 # 1-gram weights
        self.w2 = w2 # 2-grams weights
        self.w3 = w3 # 3-grams weights
        self.w4 = w4 # 4-grams weights
    
    def compute_blue_score(self, real, predicted):
        score = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score.append(sentence_bleu([sent1], sent2, 
                          weights=(self.w1, self.w2, self.w3, self.w4)))
        return score

class AVE_Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            frames: a list of Tensor image of size (C, H, W) to be normalized.
        Returns:
            a list of Tensor: a list of normalized Tensor images.
        """

        out_frames = F.normalize(frames, self.mean, self.std)
        return out_frames

class Normalize(torch.nn.Module) :
    def __init__(self, mean, std) :
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).to(device))
        self.register_buffer('std', torch.Tensor(std).to(device))
        
    def forward(self, input):
        #print(input.size()) [1, 40, 3, 128, 128] [1,1,3,16,128,128]
		# Broadcasting
        #input = input/255.0
        mean = self.mean.reshape(1, 3, 1, 1, 1)
        std = self.std.reshape(1, 3, 1, 1, 1)
        return ((input - mean) / std).to(device)
        #return ((input * std) - mean).to(device)

class DeNormalize(torch.nn.Module) :
    def __init__(self, mean, std) :
        super(DeNormalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean).to(device))
        self.register_buffer('std', torch.Tensor(std).to(device))
        
    def forward(self, input):
        # [1,3,16,128,128]
        mean = self.mean.reshape(1, 3, 1, 1, 1)
        std = self.std.reshape(1, 3, 1, 1, 1)
        return ((input + mean) / std).to(device)
    
def calculate_performance(cfg, rx_img, tx_img, real_img):

    # Get the int8 generated images
    img_gen_numpy = rx_img.detach().cpu().float().numpy()
    img_gen_numpy = (np.transpose(img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    img_gen_int8 = img_gen_numpy.astype(np.uint8)

    tx_img_gen_numpy = tx_img.detach().cpu().float().numpy()
    tx_img_gen_numpy = (np.transpose(tx_img_gen_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    tx_img_gen_int8 = tx_img_gen_numpy.astype(np.uint8)

    origin_numpy = real_img.detach().cpu().float().numpy()
    origin_numpy = (np.transpose(origin_numpy, (0, 2, 3, 1)) + 1) / 2.0 * 255.0
    origin_int8 = origin_numpy.astype(np.uint8)

    # Get the PSNR/SSIM
    diff = np.mean((np.float64(img_gen_int8) - np.float64(origin_int8))**2, (1, 2, 3))
    tx_diff = np.mean((np.float64(tx_img_gen_int8) - np.float64(origin_int8))**2, (1, 2, 3))

    rx_PSNR = 10 * np.log10((255**2) / diff)
    tx_PSNR = 10 * np.log10((255**2) / tx_diff)

    img_gen_tensor = torch.from_numpy(np.transpose(img_gen_int8, (0, 3, 1, 2))).float()
    tx_img_gen_tensor = torch.from_numpy(np.transpose(tx_img_gen_int8, (0, 3, 1, 2))).float()
    origin_tensor = torch.from_numpy(np.transpose(origin_int8, (0, 3, 1, 2))).float()

    rx_ssim_val = ssim(img_gen_tensor, origin_tensor.repeat(cfg.how_many_channel, 1, 1, 1), data_range=255, size_average=False)  # return (N,)
    tx_ssim_val = ssim(tx_img_gen_tensor, origin_tensor.repeat(cfg.how_many_channel, 1, 1, 1), data_range=255, size_average=False)  # return (N,)

    if rx_PSNR.size == 1:    
        return rx_PSNR.item(), tx_PSNR.item(), rx_ssim_val.item(), tx_ssim_val.item()
    else:
        return rx_PSNR.mean().item(), tx_PSNR.mean().item(), rx_ssim_val.mean().item(), tx_ssim_val.mean().item()

def compute_power(cfg, perturbation, size_latent):
    N = perturbation.size(0)
    pert = torch.ifft(perturbation.detach().clone(), 1)
    pert = pert.view(N, cfg.P, (size_latent+1)*(cfg.M+cfg.K), 2)
    pert_pwr = torch.mean(pert**2, (-2,-1), True)
    
    return pert_pwr

def store_image(img, save_path):
    # img : [1, 3, 128, 128]
    img = img * 255.0
    clamped_img = img.clamp(0,255) # [1,1,3,16,128,128]
    
    clamped_img = clamped_img[0]
    images_np = clamped_img.detach().cpu().numpy() 

    store_img = images_np
    store_img = np.moveaxis(store_img, 0, 2) 

    imageio.imwrite(save_path,
                np.uint8(np.round(store_img)))