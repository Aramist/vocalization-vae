#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import logging
import os
from os import path
from pathlib import Path
import time

import auraloss
import torch
from torch import optim
from torch.utils.data import DataLoader

from model import VocalizationVAE
from utils.dataloaders import VocalizationDataset


def reload_cfg(cfg_path):
    """ I'm implementing it this way because I want to be able to change hyperparams
    like learning rate while the program is still running. While inefficient, this
    is the simplest way to do so.
    """
    with open(cfg_path, 'r') as ctx:
        cfg_data = json.load(ctx)
    return cfg_data


def loss_fn(pred, target, elbo, beta=0):
    """ Evaluates the objective function.
    Parameters:
        - pred: Output of the model
        - target: Data passed into the model
        - elbo: The evidence lower bound computed by the model for this sample's posterior
        - beta: Weighting coefficient for the ELBO
    """
    # Target is initially chunked, (batch, n_blocks, block_size)
    target = target.flatten(start_dim=-2).unsqueeze(1)  # reshapes -> (batch, 1, n_samp)
    pred = pred.unsqueeze(1)  # Reshapes (batch, n_samp) -> (batch, 1, n_samp)
    # Unsqueezing the channel dimension might not be necessary, but the loss package
    # I'm using isn't well documented and I didn't feel like checking the source myself
    
    stftloss = auraloss.freq.MultiResolutionSTFTLoss()
    reconst = stftloss(pred, target)

    return reconst + elbo * beta, reconst, elbo


def train(cfg_path, logger, *, report_freq=75):
    cfg = reload_cfg(cfg_path)
    num_epochs = cfg['num_epochs']
    data_path = cfg['data_path']
    weight_save_dir = cfg['weight_save_dir']
    weight_save_interval = cfg['weight_save_interval']
    beta = cfg['beta_coeff']
    lr = cfg['learning_rate']
    latent_dim = cfg['latent_dim']

    cuda = torch.cuda.is_available()
    model = VocalizationVAE(crop_size=4096*6, latent_dim=latent_dim)  # About 200ms
    if cuda:
        model.cuda()
    
    logger.info(model.__repr__())
    logger.info(f"CUDA availability: {'yes' if cuda else 'no'}")
    
    opt = optim.Adam(model.param_groups(), lr=lr, betas=(0.9, 0.999))
    opt.param_groups[1]['lr'] = lr / 10
    dset = VocalizationDataset(data_path, model.crop_size, model.block_size)
    dloader = DataLoader(dset, batch_size=24, shuffle=True, collate_fn=dset.collate_fn)
    
    logger.info('Making save directories')
    os.makedirs(weight_save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        start_time = time.time()
        for n, data in enumerate(dloader):
            if cuda:
                data = data.cuda()
            
            opt.zero_grad()
            output, elbo = model(data)
            # Save all three values for reporting
            loss, reconst, kl = loss_fn(output, data, elbo, beta=beta)
            loss.backward()

            opt.step()

            if (n % report_freq) == (report_freq - 1):
                logger.info('Epoch {} progress: {}/{}. Last minibatch loss: {:.4e}. Reconstruction: {:.3f}. KL: {:.3f}'.format(
                    epoch+1, n+1, len(dloader),
                    loss.detach().cpu().item(),
                    reconst.detach().cpu().item(),
                    kl.detach().cpu().item()
                ))
                logger.info(torch.cuda.memory_summary(abbreviated=True))
                avg_time = '{:.1f}'.format((time.time() - start_time) / (n + 1))
                logger.info(f'Avg. time per minibatch: {avg_time}s')

            # Since the data are so large, checkpoint periodically within epochs
            if (n % weight_save_interval) == (weight_save_interval - 1):
                weights = model.state_dict()
                weight_path = path.join(weight_save_dir, f'ep{epoch+1}_mb{n+1}.pt')
                torch.save(weights, weight_path)
        # Also checkpoint at the end of every epoch
        weights = model.state_dict()
        weight_path = path.join(weight_save_dir, f'ep{epoch+1}_final.pt')
        torch.save(weights, weight_path)

        # Check config to see if hyperparams need changing
        cfg = reload_cfg(cfg_path)
        beta = cfg['beta_coeff']
        lr = cfg['learning_rate']
        logger.info("Epoch complete, updating lr to {:.3e} and beta to {:.3e}".format(lr, beta))
        opt.param_groups[0]['lr'] = lr
        opt.param_groups[1]['lr'] = lr / 10


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to model config'
    )
    args = parser.parse_args()

    if not path.exists(args.config_path):
        raise ValueError(f"Could not find config file located at: {args.config_path}")
    
    return args.config_path


if __name__ == '__main__':
    cfg_path = get_cfg()

    cfgname = Path(cfg_path).stem
    logger = logging.getLogger('myLogger')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(f'/mnt/home/atanelus/Heap/audio_vae/log_{cfgname}.log'))
    logger.info("Initializing...")
    train(cfg_path, logger)
