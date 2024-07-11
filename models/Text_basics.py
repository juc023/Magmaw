import spacy
import numpy as np
import random
import math
import time
import spacy
import torch
import torch.nn as nn
import torchtext
from torchtext.datasets import Multi30k, WMT14, IWSLT
from torchtext.data import Field, BucketIterator

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def get_size_latent(sentence, src_field):
    if isinstance(sentence, str):
        nlp = spacy.load('de_core_news_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    size_latent = len(src_indexes) * (256 // 2) // 64
    return size_latent

def prepare_text_dataset():
    SRC = Field(tokenize = tokenize_de, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True)

    TRG = Field(tokenize = tokenize_en, 
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True, 
                batch_first = True)
    
    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))


    '''
    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE,
        device = device)'''

    return SRC, TRG, train_data, test_data


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs