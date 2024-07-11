import math
import time

import torch as th
from torch import nn
from torch.nn import functional as F
from .resample import downsample2, upsample2
from .utils import capture_init
import pdb
import os
from . import channel
import numpy as np

class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden

def Channel_Normalize(x, pwr=1):
    '''
    Normalization function
    '''
    power = th.mean(x**2, (-2,-1), True)
    alpha = np.sqrt(pwr/2)/th.sqrt(power)
    return alpha*x

def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class Demucs(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.
    """
    @capture_init
    def __init__(self,
                 opt,
                 device,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.device = device
        self.checkpoints_dir = opt.checkpoints_dir
        self.speech_name = opt.speech_name
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.img_name)
        if opt.isTrain != True:
            self.load_iter = opt.load_iter
            self.epoch = opt.epoch

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        self.sp_jscc_enc = Sp_JSCC_Encoder(ENC_DROPOUT)
        self.sp_jscc_dec = Sp_JSCC_Decoder(DEC_DROPOUT)
        
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1),
                activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)
        
        self.channel = channel.OFDM_channel(opt, self.device, pwr=1)
        self.perturbation = None

    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))

        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)

        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = x.permute(2, 0, 1)
        
        # LSTM input [255, 48, 768]
        x, _ = self.lstm(x)
        # [255, 48, 768] -> [48, 255, 768] -> [48, 85, 2304]

        x = x.permute(1, 0, 2)
        '''
        Wireless
        '''
        # [48, 255, 768] -> [48, 255, 64]
        x = self.sp_jscc_enc(x)
        
        
        #self.wireless_channel(x)
        
        # [48, 255, 64] -> [48, 255, 768]
        x = self.sp_jscc_dec(x)

        # [48, 85, 2304] -> [48, 255, 768] -> [48, 768, 255]
        x = x.permute(0, 2, 1)

        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)

        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x)
        
        x = x[..., :length]
        return std * x

    def wireless_channel(self, latent, cof_in=None):
        '''
        latent: [48, 255, 64]
        '''
        latent = th.unsqueeze(latent, 1)
        N, C, H, W = latent.shape
        img_S = int(latent.shape[2] * (latent.shape[3] // 2) / self.opt.M)
        
        # Generate information about the channel when available
        if cof_in is not None:
            cof, H_true = cof_in
        elif cof_in is None and self.opt.feedforward == 'OFDM-feedback':
            cof, H_true = self.channel.sample(N)
        else:
            cof, H_true = None, None

        # Pre-coding process when the channel feedback is available
        if self.opt.feedforward == 'OFDM-feedback':
            H_true = H_true.permute(0, 1, 3, 2).contiguous().view(N, -1, latent.shape[2], latent.shape[3]).to(latent.device)
            weights = self.netP(th.cat((H_true, latent), 1))
            latent = latent*weights

        # Reshape the latents to be transmitted
        # 128, 1, 6, 2, 64 = [batch, pilot, Number of packets, 2, Number of subcarriers per symbol]
        # latent.size() = [B, 12, 8, 8] -> [B, 1, 6, 64, 2]
        self.tx = latent.view(N, self.opt.P, img_S, 2, self.opt.M).permute(0,1,2,4,3)

        # Modulation
        if (self.opt.enable_modulation == True):
            constell = th.zeros(self.opt.m_degree,2).to(latent.device)
            constell = self.create_modulation(constell, self.opt.modulation, self.opt.m_degree)    
            if (inference == True):
                self.tx = self.soft_hard_mod(self.tx, constell)
            else:     
                self.tx = self.soft_hard_mod(self.tx, constell)

        #self.draw_constellation(self.tx)

        # Transmit through the channel
        out_pilot, out_sig, self.H_true, noise_pwr, self.PAPR, normalized_pert_pwr, max_pert_pwr = self.channel(self.tx, SNR=self.opt.SNR, 
                                                                                        size_latent=img_S, 
                                                                                        perturbation=self.perturbation, 
                                                                                        cof=cof)
        self.H_true = self.H_true.to(self.device).unsqueeze(2)
        self.normalized_pert_pwr = normalized_pert_pwr
        self.max_pert_pwr = max_pert_pwr

        # TX reference frame
        tx_ref = latent.view(N, self.opt.P, img_S, 2, self.opt.M).permute(0,1,2,4,3)
        normalized_tx_ref = Channel_Normalize(tx_ref)
        tx_dec_in = normalized_tx_ref.contiguous().permute(0,1,2,4,3).contiguous().view(N, -1, H, W) #[N, 12, 8, 8]    

        pdb.set_trace()

    def save_model(self, model, iter):
        save_filename = 'iter{}.model'.format(iter)
        #save_dir = os.path.join(self.checkpoints_dir, self.speech_name)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        th.save(model.state_dict(), save_path)

    def load_model(self, model):
        load_suffix = 'iter_%d' % self.load_iter if self.load_iter > 0 else 'iter%s' % self.epoch
        load_filename = '%s.model' % (load_suffix)
        #save_dir = os.path.join(self.checkpoints_dir, self.speech_name)
        load_path = os.path.join(self.save_dir, load_filename)
        #pdb.set_trace()
        with open(load_path, 'rb') as f:
            pretrained_dict = th.load(f)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        f = str(f)
        '''
        if f.find('iter') != -1 and f.find('.model') != -1:
            st = f.find('iter') + 4
            ed = f.find('.model', st)
            return int(f[st:ed])
        else:
            return 0
        '''
'''
class Sp_JSCC_Encoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        
        self.fc1 = nn.Linear(2304, 1152)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1152, 576)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(576, 384)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(384, 256)
                
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(256)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        x = self.relu3(self.fc3(x))
        x = self.dropout(x)
        y = self.fc4(x)
        
        y = self.ln(self.dropout(y))
        
        return y

class Sp_JSCC_Decoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        
        self.fc1 = nn.Linear(256, 384)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(384, 576)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(576, 1152)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(1152, 2304)
        self.ln = nn.LayerNorm(2304)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        y = self.dropout(x)
        y = self.relu2(self.fc2(y))
        y = self.dropout(y)
        y = self.relu3(self.fc3(y))
        y = self.dropout(y)
        y = self.fc4(y)
        
        y = self.ln(self.dropout(y))
        
        return y
'''
class Sp_JSCC_Encoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        
        self.fc1 = nn.Linear(768, 384)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(384, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 128)
                
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(128)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        y = self.fc3(x)
        
        y = self.ln(self.dropout(y))
        
        return y

class Sp_JSCC_Decoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        
        self.fc1 = nn.Linear(128, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 384)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(384, 768)
        self.ln = nn.LayerNorm(768)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        y = self.dropout(x)
        y = self.relu2(self.fc2(y))
        y = self.dropout(y)
        y = self.fc3(y)
        
        y = self.ln(self.dropout(y))
        
        return y

def fast_conv(conv, x):
    """
    Faster convolution evaluation if either kernel size is 1
    or length of sequence is 1.
    """
    batch, chin, length = x.shape
    chout, chin, kernel = conv.weight.shape
    assert batch == 1
    if kernel == 1:
        x = x.view(chin, length)
        out = th.addmm(conv.bias.view(-1, 1),
                       conv.weight.view(chout, chin), x)
    elif length == kernel:
        x = x.view(chin * kernel, 1)
        out = th.addmm(conv.bias.view(-1, 1),
                       conv.weight.view(chout, chin * kernel), x)
    else:
        out = conv(x)
    return out.view(batch, chout, -1)


class DemucsStreamer:
    """
    Streaming implementation for Demucs. It supports being fed with any amount
    of audio at a time. You will get back as much audio as possible at that
    point.
    Args:
        - demucs (Demucs): Demucs model.
        - dry (float): amount of dry (e.g. input) signal to keep. 0 is maximum
            noise removal, 1 just returns the input signal. Small values > 0
            allows to limit distortions.
        - num_frames (int): number of frames to process at once. Higher values
            will increase overall latency but improve the real time factor.
        - resample_lookahead (int): extra lookahead used for the resampling.
        - resample_buffer (int): size of the buffer of previous inputs/outputs
            kept for resampling.
    """
    def __init__(self, demucs,
                 dry=0,
                 num_frames=1,
                 resample_lookahead=64,
                 resample_buffer=256):
        device = next(iter(demucs.parameters())).device
        self.demucs = demucs
        self.lstm_state = None
        self.conv_state = None
        self.dry = dry
        self.resample_lookahead = resample_lookahead
        resample_buffer = min(demucs.total_stride, resample_buffer)
        self.resample_buffer = resample_buffer
        self.frame_length = demucs.valid_length(1) + demucs.total_stride * (num_frames - 1)
        self.total_length = self.frame_length + self.resample_lookahead
        self.stride = demucs.total_stride * num_frames
        self.resample_in = th.zeros(demucs.chin, resample_buffer, device=device)
        self.resample_out = th.zeros(demucs.chin, resample_buffer, device=device)

        self.frames = 0
        self.total_time = 0
        self.variance = 0
        self.pending = th.zeros(demucs.chin, 0, device=device)

        bias = demucs.decoder[0][2].bias
        weight = demucs.decoder[0][2].weight
        chin, chout, kernel = weight.shape
        self._bias = bias.view(-1, 1).repeat(1, kernel).view(-1, 1)
        self._weight = weight.permute(1, 2, 0).contiguous()

    def reset_time_per_frame(self):
        self.total_time = 0
        self.frames = 0

    @property
    def time_per_frame(self):
        return self.total_time / self.frames

    def flush(self):
        """
        Flush remaining audio by padding it with zero and initialize the previous
        status. Call this when you have no more input and want to get back the last
        chunk of audio.
        """
        self.lstm_state = None
        self.conv_state = None
        pending_length = self.pending.shape[1]
        padding = th.zeros(self.demucs.chin, self.total_length, device=self.pending.device)
        out = self.feed(padding)
        return out[:, :pending_length]

    def feed(self, wav):
        """
        Apply the model to mix using true real time evaluation.
        Normalization is done online as is the resampling.
        """
        begin = time.time()
        demucs = self.demucs
        resample_buffer = self.resample_buffer
        stride = self.stride
        resample = demucs.resample

        if wav.dim() != 2:
            raise ValueError("input wav should be two dimensional.")
        chin, _ = wav.shape
        if chin != demucs.chin:
            raise ValueError(f"Expected {demucs.chin} channels, got {chin}")

        self.pending = th.cat([self.pending, wav], dim=1)
        outs = []
        while self.pending.shape[1] >= self.total_length:
            self.frames += 1
            frame = self.pending[:, :self.total_length]
            dry_signal = frame[:, :stride]
            if demucs.normalize:
                mono = frame.mean(0)
                variance = (mono**2).mean()
                self.variance = variance / self.frames + (1 - 1 / self.frames) * self.variance
                frame = frame / (demucs.floor + math.sqrt(self.variance))
            padded_frame = th.cat([self.resample_in, frame], dim=-1)
            self.resample_in[:] = frame[:, stride - resample_buffer:stride]
            frame = padded_frame

            if resample == 4:
                frame = upsample2(upsample2(frame))
            elif resample == 2:
                frame = upsample2(frame)
            frame = frame[:, resample * resample_buffer:]  # remove pre sampling buffer
            frame = frame[:, :resample * self.frame_length]  # remove extra samples after window

            out, extra = self._separate_frame(frame)
            padded_out = th.cat([self.resample_out, out, extra], 1)
            self.resample_out[:] = out[:, -resample_buffer:]
            if resample == 4:
                out = downsample2(downsample2(padded_out))
            elif resample == 2:
                out = downsample2(padded_out)
            else:
                out = padded_out

            out = out[:, resample_buffer // resample:]
            out = out[:, :stride]

            if demucs.normalize:
                out *= math.sqrt(self.variance)
            out = self.dry * dry_signal + (1 - self.dry) * out
            outs.append(out)
            self.pending = self.pending[:, stride:]

        self.total_time += time.time() - begin
        if outs:
            out = th.cat(outs, 1)
        else:
            out = th.zeros(chin, 0, device=wav.device)
        return out

    def _separate_frame(self, frame):
        demucs = self.demucs
        skips = []
        next_state = []
        first = self.conv_state is None
        stride = self.stride * demucs.resample
        x = frame[None]
        for idx, encode in enumerate(demucs.encoder):
            stride //= demucs.stride
            length = x.shape[2]
            if idx == demucs.depth - 1:
                # This is sligthly faster for the last conv
                x = fast_conv(encode[0], x)
                x = encode[1](x)
                x = fast_conv(encode[2], x)
                x = encode[3](x)
            else:
                if not first:
                    prev = self.conv_state.pop(0)
                    prev = prev[..., stride:]
                    tgt = (length - demucs.kernel_size) // demucs.stride + 1
                    missing = tgt - prev.shape[-1]
                    offset = length - demucs.kernel_size - demucs.stride * (missing - 1)
                    x = x[..., offset:]
                x = encode[1](encode[0](x))
                x = fast_conv(encode[2], x)
                x = encode[3](x)
                if not first:
                    x = th.cat([prev, x], -1)
                next_state.append(x)
            skips.append(x)

        x = x.permute(2, 0, 1)
        x, self.lstm_state = demucs.lstm(x, self.lstm_state)
        x = x.permute(1, 2, 0)
        # In the following, x contains only correct samples, i.e. the one
        # for which each time position is covered by two window of the upper layer.
        # extra contains extra samples to the right, and is used only as a
        # better padding for the online resampling.
        extra = None
        for idx, decode in enumerate(demucs.decoder):
            skip = skips.pop(-1)
            x += skip[..., :x.shape[-1]]
            x = fast_conv(decode[0], x)
            x = decode[1](x)

            if extra is not None:
                skip = skip[..., x.shape[-1]:]
                extra += skip[..., :extra.shape[-1]]
                extra = decode[2](decode[1](decode[0](extra)))
            x = decode[2](x)
            next_state.append(x[..., -demucs.stride:] - decode[2].bias.view(-1, 1))
            if extra is None:
                extra = x[..., -demucs.stride:]
            else:
                extra[..., :demucs.stride] += next_state[-1]
            x = x[..., :-demucs.stride]

            if not first:
                prev = self.conv_state.pop(0)
                x[..., :demucs.stride] += prev
            if idx != demucs.depth - 1:
                x = decode[3](x)
                extra = decode[3](extra)
        self.conv_state = next_state
        return x[0], extra[0]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        "denoiser.demucs",
        description="Benchmark the streaming Demucs implementation, "
                    "as well as checking the delta with the offline implementation.")
    parser.add_argument("--depth", default=5, type=int)
    parser.add_argument("--resample", default=4, type=int)
    parser.add_argument("--hidden", default=48, type=int)
    parser.add_argument("--sample_rate", default=16000, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-t", "--num_threads", type=int)
    parser.add_argument("-f", "--num_frames", type=int, default=1)
    args = parser.parse_args()
    if args.num_threads:
        th.set_num_threads(args.num_threads)
    sr = args.sample_rate
    sr_ms = sr / 1000
    demucs = Demucs(depth=args.depth, hidden=args.hidden, resample=args.resample).to(args.device)
    x = th.randn(1, int(sr * 4)).to(args.device)
    out = demucs(x[None])[0]
    streamer = DemucsStreamer(demucs, num_frames=args.num_frames)
    out_rt = []
    frame_size = streamer.total_length
    with th.no_grad():
        while x.shape[1] > 0:
            out_rt.append(streamer.feed(x[:, :frame_size]))
            x = x[:, frame_size:]
            frame_size = streamer.demucs.total_stride
    out_rt.append(streamer.flush())
    out_rt = th.cat(out_rt, 1)
    model_size = sum(p.numel() for p in demucs.parameters()) * 4 / 2**20
    initial_lag = streamer.total_length / sr_ms
    tpf = 1000 * streamer.time_per_frame
    print(f"model size: {model_size:.1f}MB, ", end='')
    print(f"delta batch/streaming: {th.norm(out - out_rt) / th.norm(out):.2%}")
    print(f"initial lag: {initial_lag:.1f}ms, ", end='')
    print(f"stride: {streamer.stride * args.num_frames / sr_ms:.1f}ms")
    print(f"time per frame: {tpf:.1f}ms, ", end='')
    print(f"RTF: {((1000 * streamer.time_per_frame) / (streamer.stride / sr_ms)):.2f}")
    print(f"Total lag with computation: {initial_lag + tpf:.1f}ms")


if __name__ == '__main__':
    main()