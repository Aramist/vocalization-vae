#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
import os
from os import path
from typing import Callable, Optional, Union

import auraloss
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from utils import *


def crop_n_block_n_scale(data: torch.Tensor, crop_size: int, block_size: int):
    if crop_size % block_size != 0:
        raise ValueError('block_size should be a factor of crop_size')
    if data.shape[1] < crop_size:
        pad = torch.zeros((data.shape[0], crop_size), dtype=data.dtype, device=data.device)
        dmin, dmax = data.min(dim=1, keepdims=True)[0], data.max(dim=1, keepdims=True)[0]
        data = (data - dmin) / (dmax-dmin) - 0.5  # between -0.5 and 0.5
        # scale before padding so pad values remain zero
        pad[:, :data.shape[1]] = data
        data = pad
    else:
        data = data[:, :crop_size]
        # Scale after cropping so the output range isn't affected by the crop
        dmin, dmax = data.min(dim=1, keepdims=True)[0], data.max(dim=1, keepdims=True)[0]
        data = (data - dmin) / (dmax-dmin) - 0.5  # between -0.5 and 0.5
    return data.reshape(data.shape[0], -1, block_size)


# In[3]:


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_size = 16
        self.d_model = 512
        
        self.num_heads = 8
        self.latent_dim = 32
        
        self.crop_size = 4096
        self.max_seq = self.crop_size // self.block_size
        
        self.in_encoding = LearnedEncoding(d_model=self.d_model, max_seq_len=self.max_seq)
        self.out_encoding = LearnedEncoding(d_model=self.d_model, max_seq_len=self.max_seq)
        
        
        self.data_encoding = nn.Linear(self.block_size, self.d_model)
        self.encoder = SparseTransformerEncoder(
            SparseTransformerEncoderLayer(
                self.d_model,
                self.num_heads,
                block_size=8,
                n_global=4,
                n_window=11,
                n_random=6,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            ),
            4
        )
        
        """self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self.d_model, self.num_heads, batch_first=True
            ),
            4
        )"""
        
        # Encoder will produce two vectors of dim latent_dim, which encode the mean and log(variance) of the posterior
        # Might try a custom implementation of teh decoder that scraps cross-attention weights since all the
        # attention targets will be the same
        self.post_mean = nn.Linear(self.d_model, self.latent_dim)
        self.post_logvar = nn.Linear(self.d_model, self.latent_dim)
        
        self.expansion = nn.Linear(self.latent_dim, self.max_seq * 2)
        
        # Basically an outer product
        self.conv_expansion = nn.Conv1d(
            in_channels=2,
            out_channels=self.d_model,
            kernel_size=3,
            padding='same'
        )
        
        self.decoder_blocks = nn.ModuleList([
            SparseTransformerEncoder(
                SparseTransformerEncoderLayer(
                    self.d_model,
                    self.num_heads,
                    block_size=8,
                    n_global=4,
                    n_window=11,
                    n_random=6,
                    dim_feedforward=2048,
                    dropout=0.1,
                    batch_first=True
                ),
                6
            )
         ])
        
        
        # Somewhat works after 2500 epochs
        """self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    self.d_model,
                    self.num_heads,
                    batch_first=True
                ),
                6
            ),
        ])"""
        
        self.to_seq = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=16,
            kernel_size=1
        )
    
    def collate_fn(self, batch):
        batch = [torch.from_numpy(b).float() for b in batch]
        batch = nn.utils.rnn.pad_sequence(batch, batch_first=True)
        batch = crop_n_block_n_scale(batch, self.crop_size - self.block_size, self.block_size)
        return batch
        

    def _clip_gradients(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
    
    def encode(self, x):
        batched = x.dim() == 3
        
        # Initial shape: (blocks, block_size)
        embed_out = self.data_encoding(x)  # Outputs (blocks, d_model)
        cls_token = torch.zeros((1, self.d_model), device=x.device)
        if batched:
            cls_token = cls_token.unsqueeze(0).expand(x.shape[0], -1, -1)
        transformer_in = torch.cat([cls_token, embed_out], dim=-2)
        transformer_in = self.in_encoding(transformer_in.unsqueeze(0) if not batched else transformer_in)
        transformer_out = self.encoder(transformer_in)[:, 0, :]
        mean, log_var = self.post_mean(transformer_out), self.post_logvar(transformer_out)
        return mean, log_var
    
    @classmethod
    def sample(cls, mean, log_var):
        # Independant normals
        cov = torch.diag_embed(torch.exp(log_var))
        dist = MultivariateNormal(mean, cov)
        return dist.rsample()
    
    @classmethod
    def elbo(cls, mean, log_var):
        """ Evidence lower bound, mean-reduced across batch dim.
        https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py
        """
        elbo = -0.5 * (1 + log_var - mean**2 - log_var.exp()).sum(dim=1)
        return elbo.mean(dim=0)
    
    def decode(self, z):
        # Expected input shape: (1, latent_dim)
        seq = self.expansion(z).unsqueeze(1)  # (1, latent_dim) -> (1, 1, max_seq_len * 2)
        seq = seq.reshape(seq.shape[0], 2, -1)  # (1, 1, max_seq_len * 2) -> (1, 2, max_seq_len)
        seq = self.conv_expansion(seq).transpose(1, 2)  # (1, 1, max_seq_len*2) -> (1, max_seq_len*2, 1)
        
        decode = self.out_encoding(seq)
        for decoder in self.decoder_blocks:
            decode = decoder(decode)
        decode = decode.transpose(1, 2)  # (1, seq, feat) -> (1, feat, seq)
        return self.to_seq(decode).transpose(-1, -2).flatten(start_dim=-2)
        
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = AutoEncoder.sample(mean, logvar)
        decoded = self.decode(z)
        return decoded, AutoEncoder.elbo(mean, logvar)


# In[4]:


def loss_fn(pred, target, elbo, beta=0):
    target = target.flatten(start_dim=-2).unsqueeze(1)  # (n_blocks, block_size) -> (n_samp,)
    pred = pred.unsqueeze(1)
    
    stftloss = auraloss.freq.MultiResolutionSTFTLoss()
    reconst = stftloss(pred, target)
    # stft_target = torch.stft(target, n_fft=512, return_complex=True)
    # stft_pred = torch.stft(pred, n_fft=512, return_complex=True)
    # diff = stft_target - stft_pred
    # diff = diff * torch.conj(diff)
    # reconst = diff.real.mean(dim=(-1, -2)).mean()
    # reconst = ((target - pred[:,:target.shape[1]])**2).mean(dim=-1).mean()
    return reconst + elbo * beta, reconst, elbo


# In[5]:


model = AutoEncoder().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
dset = UnlabeledDataset('/mnt/ceph/users/atanelus/unlabeled_vocalizations/c3/merged.h5')
dloader = DataLoader(dset, batch_size=64, shuffle=True, collate_fn = model.collate_fn)


# In[ ]:


report_interval = 1
for epoch in range(3000):
    for n, batch in enumerate(dloader):
        batch = batch.cuda()
        
        optimizer.zero_grad()
        output, elbo = model(batch)
        loss, reconst, kl = loss_fn(output, batch, elbo, beta=0.01)
        loss.backward()
        # for name, param in model.named_parameters():
        #     print(name, param.grad)
        # model._clip_gradients()
        optimizer.step()
        
        if (n % report_interval) == (report_interval - 1):
            print('Epoch {} progress: {}/{}. Last minibatch loss: {:.4e}. Reconstruction: {:.3f}. KL: {:.3f}'.format(
                epoch+1, n+1, len(dloader),
                loss.detach().cpu().item(),
                reconst.detach().cpu().item(),
                kl.detach().cpu().item()
            ))
            print(torch.cuda.memory_summary(abbreviated=True))
        
        if (n % 2000) == 1999:
            weights = model.state_dict()
            path = f'/mnt/ceph/users/atanelus/vae/ep{epoch+1}_mb{n+1}.pt'
            torch.save(weights, path)
    weights = model.state_dict()
    path = f'/mnt/ceph/users/atanelus/vae/ep{epoch+1}_final.pt'
    torch.save(weights, path)


# In[ ]:


plt.specgram(output[0].detach().cpu().numpy())
plt.show()
plt.specgram(batch[0].flatten().cpu().numpy())
plt.show()


# In[ ]:





# In[ ]:




