import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .base_model import BaseModel
from . import channel
from .JSCCOFDM_model import Channel_Normalize
import pdb
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def text_save_model(model, checkpoints_dir, iter):
    save_filename = 'iter_{}.model'.format(iter)
    save_path = os.path.join(checkpoints_dir, save_filename)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    torch.save(model.state_dict(), save_path)

def text_load_model(model, checkpoints_dir, iter):
    load_filename = 'iter_{}.model'.format(iter)
    load_path = os.path.join(checkpoints_dir, load_filename)
    model.load_state_dict(torch.load(load_path))

class Text_Transmission_net(BaseModel):
    def __init__(self, opt, INPUT_DIM, OUTPUT_DIM, SRC_PAD_IDX, TRG_PAD_IDX, device):
        BaseModel.__init__(self, opt)
        
        HID_DIM = 256
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 8
        DEC_HEADS = 8
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1

        self.src_pad_idx = SRC_PAD_IDX
        self.trg_pad_idx = TRG_PAD_IDX
        self.device = device
        self.tx_jscc_enc = Tx_JSCC_Encoder(ENC_DROPOUT).to(device)
        self.tx_jscc_dec = Tx_JSCC_Decoder(DEC_DROPOUT).to(device)

        self.encoder = Encoder(INPUT_DIM, 
                    HID_DIM, 
                    ENC_LAYERS, 
                    ENC_HEADS, 
                    ENC_PF_DIM, 
                    ENC_DROPOUT, 
                    device)

        self.decoder = Decoder(OUTPUT_DIM, 
                    HID_DIM, 
                    DEC_LAYERS, 
                    DEC_HEADS, 
                    DEC_PF_DIM, 
                    DEC_DROPOUT, 
                    device)
        
        self.perturbation = None
        self.channel = channel.OFDM_channel(opt, self.device, pwr=1)
        
        #self.Text_Transformer = Text_Transformer(enc, dec, device)

    def list_to_tensor(self, sentence, src_field, device):
        if isinstance(sentence, str):
            nlp = spacy.load('de_core_news_sm')
            tokens = [token.text.lower() for token in nlp(sentence)]
        else:
            tokens = [token.lower() for token in sentence]

        tokens = [src_field.init_token] + tokens + [src_field.eos_token]
            
        src_indexes = [src_field.vocab.stoi[token] for token in tokens]

        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        
        return src_tensor

    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg, cof=None):        
        #src = [batch size, src len]
        #trg = [batch size, trg len]        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #enc_src = [batch size, src len, hid dim] = [batch size, src len, 256]
        enc_src = self.encoder(src, src_mask)
        
        tx_enc_src = self.tx_jscc_enc(enc_src)
        
        # Reshape the latents to be transmitted
        # 128, 1, 6, 2, 64 = [batch, pilot, Number of packets, 2, Number of subcarriers per symbol]
        # Example) latent.size() = [B, 1, src len, 256] -> [B, 1, ?, 64, 2]
        N = tx_enc_src.size(0)
        SL = tx_enc_src.size(1)
        SC = tx_enc_src.size(2)
        
        size_latent = SL * (SC // 2) // self.opt.M
        tx_enc_src = torch.unsqueeze(tx_enc_src, 1)
        
        self.tx = tx_enc_src.view(N, self.opt.P,  size_latent, 2, self.opt.M).permute(0,1,2,4,3)

        # Transmit through the channel
        out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR, normalized_pert_pwr, max_pert_pwr = \
            self.channel(self.tx, SNR=self.opt.SNR, size_latent=size_latent, perturbation=self.perturbation, cof=cof)
        self.normalized_pert_pwr = normalized_pert_pwr
        self.H_true = self.H_true.to(self.device).unsqueeze(2)

        #N, C, H, W = latent.shape

        '''
        normalized_tx_ref = Channel_Normalize(self.tx)
        tx_enc_src = normalized_tx_ref.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, SL, SC) #[N, 12, 8, 8]  
        '''
        
        # Receiver side
        self.H_est_MMSE = self.channel_estimation(out_pilot, noise_pwr)
        self.H_est = self.H_est_MMSE
        rx = self.equalization(self.H_est, out_sig, noise_pwr) # [N, 1, 6, 64, 2]
        tx_enc_src = rx.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, SL, SC) #[N, 12, 8, 8]
        
        rx_enc_src = self.tx_jscc_dec(tx_enc_src)

        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]     
        output, attention = self.decoder(trg, rx_enc_src, trg_mask, src_mask)
        
        return output, attention

    def adv_train(self, sentence, ref_input, src_field, trg_field, device, max_len = 50, cof=None):
        src_tensor = self.list_to_tensor(sentence, src_field, device)
        #src_mask = self.make_src_mask(src_tensor)
        trg_tensor = self.list_to_tensor(ref_input, trg_field, device)
        #trg_mask = self.make_trg_mask(trg_tensor)
        
        output, _ = self.forward(src_tensor, trg_tensor[:,:-1])
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_tensor = trg_tensor[:,1:].contiguous().view(-1)
        
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
        loss = -self.criterion(output, trg_tensor)
        
        self.loss_P = loss
        
    def inference(self, sentence, src_field, trg_field, device, max_len = 50, cof=None):

        src_tensor = self.list_to_tensor(sentence, src_field, device)
        src_mask = self.make_src_mask(src_tensor)
        enc_src = self.encoder(src_tensor, src_mask)

        '''
        Wireless Transmission
        '''
        tx_enc_src = self.tx_jscc_enc(enc_src)
        
        # Reshape the latents to be transmitted
        # 128, 1, 6, 2, 64 = [batch, pilot, Number of packets, 2, Number of subcarriers per symbol]
        # Example) latent.size() = [B, 1, src len, 256] -> [B, 1, ?, 64, 2]
        N = tx_enc_src.size(0)
        SL = tx_enc_src.size(1)
        SC = tx_enc_src.size(2)

        size_latent = SL * (SC // 2) // self.opt.M
        tx_enc_src = torch.unsqueeze(tx_enc_src, 1)
        
        self.tx = tx_enc_src.view(N, self.opt.P,  size_latent, 2, self.opt.M).permute(0,1,2,4,3)

        # Transmit through the channel
        out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR, normalized_pert_pwr, max_pert_pwr = \
            self.channel(self.tx, SNR=self.opt.SNR, size_latent=size_latent, perturbation=self.perturbation, cof=cof)
        self.normalized_pert_pwr = normalized_pert_pwr
        self.H_true = self.H_true.to(self.device).unsqueeze(2)

        #N, C, H, W = latent.shape

        '''
        normalized_tx_ref = Channel_Normalize(self.tx)
        tx_enc_src = normalized_tx_ref.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, SL, SC) #[N, 12, 8, 8]  
        '''
        
        # Receiver side
        self.H_est_MMSE = self.channel_estimation(out_pilot, noise_pwr)
        self.H_est = self.H_est_MMSE
        rx = self.equalization(self.H_est, out_sig, noise_pwr) # [N, 1, 6, 64, 2]
        tx_enc_src = rx.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, SL, SC) #[N, 12, 8, 8]
        
        rx_enc_src = self.tx_jscc_dec(tx_enc_src)
        '''
        End
        '''
        
        trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
        '''
        trg_indexes = [2]
        '''
        for i in range(max_len):
            '''
            trg_tensor =
            tensor([[2]], device='cuda:0'), tensor([[  2, 731]], device='cuda:0')
            '''
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = self.make_trg_mask(trg_tensor)
            
            output, attention = self.decoder(trg_tensor, rx_enc_src, trg_mask, src_mask)

            pred_token = output.argmax(2)[:,-1].item()
            
            trg_indexes.append(pred_token)

            if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
                break

        self.output = output

        trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
        
        return trg_tokens[1:], attention

    def channel_estimation(self, out_pilot, noise_pwr):
        return channel.LMMSE_channel_est(self.channel.pilot, out_pilot, self.opt.M*noise_pwr)

    def equalization(self, H_est, out_sig, noise_pwr):
        return channel.MMSE_equalization(H_est, out_sig, self.opt.M*noise_pwr)

    def set_optimizer_perturbation(self, wireless_perturbation):
        self.perturbation = wireless_perturbation
        self.optimizer_P = torch.optim.Adam([self.perturbation], lr=self.opt.attack_lr)

    def set_criterion(self, TRG_PAD_IDX):
        self.criterion =nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    def backward_P(self, TRG, ref_input, device):
        self.optimizer_P.zero_grad()
        self.loss_P.backward()

        return self.perturbation, self.loss_P

class Tx_JSCC_Encoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        
        self.fc1 = nn.Linear(256, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(256)
        
    def forward(self, x):
        y = self.relu1(self.fc1(x))
        y = self.dropout(y)
        y = self.fc2(y)
        
        y = self.ln(x + self.dropout(y))
        
        return y

class Tx_JSCC_Decoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        
        self.fc1 = nn.Linear(256, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 256)
        self.ln = nn.LayerNorm(256)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        y = self.relu1(self.fc1(x))
        y = self.dropout(y)
        y = self.relu2(self.fc2(y))
        y = self.dropout(y)
        y = self.fc3(y)
        
        y = self.ln(x + self.dropout(y))
        
        return y

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
                
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        
        #dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        #positionwise feedforward
        _src = self.positionwise_feedforward(src)
        
        #dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        #src = [batch size, src len, hid dim]
        
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, hid dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, hid dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention


class Text_Transformer(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        self.tx_jscc_enc = Tx_JSCC_Encoder().to(device)
        self.tx_jscc_dec = Tx_JSCC_Decoder().to(device)
        
    def forward(self, src, trg, src_mask, trg_mask):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim] = [batch size, src len, 256]
        tx_enc_src = self.tx_jscc_enc(enc_src)
        rx_enc_src = self.tx_jscc_dec(tx_enc_src)
                
        output, attention = self.decoder(trg, rx_enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention