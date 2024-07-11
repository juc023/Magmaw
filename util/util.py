"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os

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

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution
    Returns:
        torch.Tensor: 1D kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size//2

    g = torch.exp(-(coords**2) / (2*sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blured
        window (torch.Tensor): 1-D gauss kernel
    Returns:
        torch.Tensor: blured tensors
    """

    N, C, H, W = input.shape
    out = F.conv2d(input, win, stride=1, padding=0, groups=C)
    out = F.conv2d(out, win.transpose(2, 3), stride=1, padding=0, groups=C)
    return out


def _ssim(X, Y, win, data_range=255, size_average=True, full=False):
    r""" Calculate ssim index for X and Y
    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    K1 = 0.01
    K2 = 0.03
    batch, channel, height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range)**2
    C2 = (K2 * data_range)**2

    win = win.to(X.device, dtype=X.dtype)

    #print(X.size())
    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * ( gaussian_filter(X * X, win) - mu1_sq )
    sigma2_sq = compensation * ( gaussian_filter(Y * Y, win) - mu2_sq )
    sigma12   = compensation * ( gaussian_filter(X * Y, win) - mu1_mu2 )

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if size_average:
        ssim_val = ssim_map.mean()
        cs = cs_map.mean()
    else:
        ssim_val = ssim_map.mean(-1).mean(-1).mean(-1)  # reduce along CHW
        cs = cs_map.mean(-1).mean(-1).mean(-1)

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
    Returns:
        torch.Tensor: ssim results
    """

    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    ssim_val, cs = _ssim(X, Y,
                         win=win,
                         data_range=data_range,
                         size_average=False,
                         full=True)
    if size_average:
        ssim_val = ssim_val.mean()
        cs = cs.mean()

    if full:
        return ssim_val, cs
    else:
        return ssim_val


def ms_ssim(X, Y, win_size=11, win_sigma=1.5, win=None, data_range=255, size_average=True, full=False, weights=None):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        full (bool, optional): return sc or not
        weights (list, optional): weights for different levels
    Returns:
        torch.Tensor: ms-ssim results
    """
    if len(X.shape) != 4:
        raise ValueError('Input images must 4-d tensor.')

    if not X.type() == Y.type():
        raise ValueError('Input images must have the same dtype.')

    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if weights is None:
        weights = torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(X.device, dtype=X.dtype)

    win_sigma = win_sigma
    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat(X.shape[1], 1, 1, 1)
    else:
        win_size = win.shape[-1]

    levels = weights.shape[0]
    mcs = []
    for _ in range(levels):
        ssim_val, cs = _ssim(X, Y,
                             win=win,
                             data_range=data_range,
                             size_average=False,
                             full=True)
        mcs.append(cs)

        padding = (X.shape[2] % 2, X.shape[3] % 2)
        X = F.avg_pool2d(X, kernel_size=2, padding=padding)
        Y = F.avg_pool2d(Y, kernel_size=2, padding=padding)

    mcs = torch.stack(mcs, dim=0)  # mcs, (level, batch)
    # weights, (level)
    msssim_val = torch.prod((mcs[:-1] ** weights[:-1].unsqueeze(1))
                            * (ssim_val ** weights[-1]), dim=0)  # (batch, )

    if size_average:
        msssim_val = msssim_val.mean()
    return msssim_val


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3):
        r""" class for ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
        """

        super(SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range

    def forward(self, X, Y):
        return ssim(X, Y, win=self.win, data_range=self.data_range, size_average=self.size_average)


class MS_SSIM(torch.nn.Module):
    def __init__(self, win_size=11, win_sigma=1.5, data_range=None, size_average=True, channel=3, weights=None):
        r""" class for ms-ssim
        Args:
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
        """

        super(MS_SSIM, self).__init__()
        self.win = _fspecial_gauss_1d(
            win_size, win_sigma).repeat(channel, 1, 1, 1)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights

    def forward(self, X, Y):
        return ms_ssim(X, Y, win=self.win, size_average=self.size_average, data_range=self.data_range, weights=self.weights)