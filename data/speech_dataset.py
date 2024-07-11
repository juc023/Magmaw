""" Load and preprocess data.
"""
import torch
#import torchaudio
import os
import pandas as pd
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader
import pdb
import pickle



class Edinburgh_dataset(Dataset):
    def __init__(self, data_dir, is_train, transform=None):
        if is_train == True:
            with open(data_dir + 'train_data.pickle', 'rb') as f:
                train_data = pickle.load(f)
        else:
            with open(data_dir + 'test_data.pickle', 'rb') as f:
                train_data = pickle.load(f)            
        self.data = train_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.data[idx].astype('float32') / 2**15
        
        if self.transform:
            x = self.transform(x)

        return x


def Load_Speech2(wav_file, window_size):
        
    sr, wav_samples = wavfile.read(wav_file)
    pdb.set_trace()
    if sr != 8000:
        print(sr)
        raise ValueError("Sampling rate is expected to be 8kHz!")
        
    assert wav_samples.ndim == 1, "check the size of wav_data"
    num_samples = wav_samples.shape[0]
    if num_samples > window_size:
        num_slices = num_samples//window_size+1
        wav_samples = np.concatenate((wav_samples, wav_samples), axis=0)
        wav_samples = wav_samples[0:window_size*num_slices]
        
        wav_slices = np.reshape(wav_samples, newshape=(num_slices, window_size))
        for wav_slice in wav_slices:
            if np.mean(np.abs(wav_slice)/2**15) < 0.015:
                num_slices -= 1
            else:
                wav_bytes = wav_slice.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={"wav_raw": bytes_feature(wav_bytes)}))
                tfrecords_file.write(example.SerializeToString())    
    else:
        num_slices = 1
        while wav_samples.shape[0] < window_size:
            wav_samples = np.concatenate((wav_samples, wav_samples), axis=0)
        
        wav_slice = wav_samples[0:window_size]
        if np.mean( np.abs(wav_slice)/2**15) < 0.015:
            num_slices -= 1
        else:
            wav_bytes = wav_slice.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={"wav_raw": bytes_feature(wav_bytes)}))
            tfrecords_file.write(example.SerializeToString())
    
    return num_slices
