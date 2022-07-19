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
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from utils.dataloaders import ClassificationDataset, LocalizationDataset
from utils.localizer import LocalizerVAE
from utils.model import VocalizationVAE


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
    # Target is initially chunked, (batch, n_blocks, block_size * n_channels)
    batch_size = target.shape[0]
    seq_len = pred.shape[1]
    target = target.reshape(batch_size, seq_len, 4)  # Now (batch, sequence, channels)
    # Looking at the source code, auraloss expects the sequence dim to be at position -1
    target = target.transpose(-1, -2).reshape(batch_size * 4, seq_len).contiguous()  # Now (batch x channels, sequence) and contiguous
    pred = pred.transpose(-1, -2).reshape(batch_size * 4, seq_len).contiguous()
    
    stftloss = auraloss.freq.MultiResolutionSTFTLoss(fft_sizes=[512, 256, 128], hop_sizes=[64, 32, 2], win_lengths=[512, 256, 128])
    reconst = stftloss(pred, target)

    return reconst + elbo * beta, reconst, elbo


def mse_loss(pred, target):
    return ((target - pred)**2).sum(dim=-1).mean()


def pred_loss_cm(pred, target, arena_dims):
    scale_factor = np.array([arena_dims[0] / 2, arena_dims[1] / 2]).reshape(1, 2)
    pred_cm = pred.detach().cpu().numpy() * scale_factor
    target_cm = target.cpu().numpy() * scale_factor
    return np.sqrt(((target_cm - pred_cm)**2).sum(axis=-1)).mean() / 10  # mm to cm


def eval(cfg_path, logger):
    cfg = reload_cfg(cfg_path)
    data_path = cfg['data_path']
    latent_dim = cfg['latent_dim']

    output_path = data_path[:-3] + "_preds.npy"
    gt_path = data_path[:-3] + "_gt.npy"

    cuda = torch.cuda.is_available()
    model = LocalizerVAE(crop_size=512, block_size=4, latent_dim=latent_dim, n_class_layers=cfg['n_layers'])
    if cuda:
        model.cuda()
    model.eval()
    
    pretrained = torch.load(cfg['pretrained_weights_path'])
    model.load_state_dict(pretrained, strict=False)
    np.save('weight.npy', model.classification_layers[0].weight.detach().cpu().numpy())
    
    logger.info(model.__repr__())
    logger.info(f"CUDA availability: {'yes' if cuda else 'no'}")
    logger.info(f"Running in eval mode for dataset {data_path}")
    dset = ClassificationDataset(data_path, model.crop_size, model.block_size, cfg['arena_width'], cfg['arena_length'])
    dloader = DataLoader(dset, batch_size=64, shuffle=False)
    
    output_list = []
    gt_labels = []

    scale_factor = np.array([cfg['arena_width'], cfg['arena_length']]).reshape(1, 2) / 2 / 10  # to cm

    with torch.no_grad():
        for n, (data, label) in enumerate(dloader):
            if cuda:
                data = data.cuda()
                # label = label.cuda()
            pred, pred_var = model(data, eval_mode=True)
            pred = pred.cpu().numpy() * scale_factor
            pred_var = pred_var.cpu().numpy() * (scale_factor ** 2)  # Var(aX) = a**2 Var(X)
            print(scale_factor, scale_factor ** 2, pred_var[0])
            cat_outputs = np.concatenate([pred, pred_var], axis=1)
            output_list.append(cat_outputs)
            gt_labels.append(label.cpu().numpy() * scale_factor)

            if (n + 1) % 5 == 4:
                logger.info(f'Progress: minibatch {n + 1} of {len(dloader)}')
    cat_outputs = np.concatenate(output_list, axis=0)
    gts = np.concatenate(gt_labels, axis=0)
    np.save(output_path, cat_outputs)
    np.save(gt_path, gts)


def train(cfg_path, logger, *, report_freq=25):
    cfg = reload_cfg(cfg_path)
    num_epochs = cfg['num_epochs']
    data_path = cfg['data_path']
    weight_save_dir = cfg['weight_save_dir']
    weight_save_interval = cfg['weight_save_interval']
    beta = cfg['beta_coeff']
    lr = cfg['learning_rate']
    latent_dim = cfg['latent_dim']
    pred_loss_weight = cfg['gamma_coeff']

    cuda = torch.cuda.is_available()
    logger.info(f"CUDA availability: {'yes' if cuda else 'no'}")
    if cfg['train_classification']:
        model = LocalizerVAE(crop_size=512, block_size=4, latent_dim=latent_dim, n_class_layers=cfg['n_layers'])
    else:
        model = VocalizationVAE(crop_size=512, block_size=4,latent_dim=latent_dim)  # About 32ms
    if cuda:
        model.cuda()
    
    if 'pretrained_weights_path' in cfg:
        pretrained = torch.load(cfg['pretrained_weights_path'])
        model.load_state_dict(pretrained, strict=False)
    
    logger.info(model.__repr__())
    logger.info(f"CUDA availability: {'yes' if cuda else 'no'}")
    
    opt = optim.Adam(model.param_groups(), lr=lr, betas=(0.9, 0.99))
    if cfg['train_classification']:
        dset = ClassificationDataset(data_path, model.crop_size, model.block_size, cfg['arena_width'], cfg['arena_length'])
    else:
        dset = LocalizationDataset(data_path, model.crop_size, model.block_size)
    dloader = DataLoader(dset, batch_size=64, shuffle=True)
    
    logger.info('Making save directories')
    os.makedirs(weight_save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}')
        start_time = time.time()
        for n, data in enumerate(dloader):
            if isinstance(data, tuple) or isinstance(data, list):
                data, label = data
            else:
                label = None
            if cuda:
                data = data.cuda()
                if label is not None:
                    label = label.cuda()
            
            opt.zero_grad()
            output = model(data)
            if label is None:
                output, elbo = output
                # Save all three values for reporting
                loss, reconst, kl = loss_fn(output, data, elbo, beta=beta)
            else:
                output, elbo, pred = output
                loss, reconst, kl = loss_fn(output, data, elbo, beta=beta)
                pred_loss = mse_loss(pred, label)
                loss = loss + pred_loss_weight * pred_loss
            loss.backward()

            opt.step()

            if (n % report_freq) == (report_freq - 1):
                pred_loss_converted = pred_loss_cm(pred, label, (cfg['arena_width'], cfg['arena_length']))
                logger.info('Epoch {} progress: {}/{}. Last minibatch loss: {:.4e}. Reconstruction: {:.3f}. KL: {:.3f}. Pred: {:.3f}cm'.format(
                    epoch+1, n+1, len(dloader),
                    loss.detach().cpu().item(),
                    reconst.detach().cpu().item(),
                    kl.detach().cpu().item(),
                    pred_loss_converted
                ))
                logger.info(torch.cuda.memory_summary(abbreviated=True))
                avg_time = '{:.1f}'.format((time.time() - start_time) / (n + 1))
                logger.info(f'Avg. time er minibatch: {avg_time}s')

            # Since the data are so large, checkpoint periodically within epochs
            if (n % weight_save_interval) == (weight_save_interval - 1):
                weights = model.state_dict()
                weight_path = path.join(weight_save_dir, 'ep{:0>4d}_mb{:0>4d}.pt'.format(epoch + 1, n + 1))
                torch.save(weights, weight_path)
        # Also checkpoint at the end of every epoch
        weights = model.state_dict()
        weight_path = path.join(weight_save_dir, 'ep{:0>4d}_final.pt'.format(epoch + 1))
        torch.save(weights, weight_path)

        # Check config to see if hyperparams need changing
        cfg = reload_cfg(cfg_path)
        beta = cfg['beta_coeff']
        lr = cfg['learning_rate']
        pred_loss_weight = cfg['gamma_coeff']
        logger.info("Epoch complete, updating lr to {:.3e}, beta to {:.3e}, and gamma to {:.3e}".format(lr, beta, pred_loss_weight))
        for group in opt.param_groups:
            group['lr'] = lr


def get_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config_path',
        type=str,
        help='Path to model config'
    )

    parser.add_argument(
        '--eval',
        action='store_true',
        help='Run the model in eval mode.'
    )
    args = parser.parse_args()

    if not path.exists(args.config_path):
        raise ValueError(f"Could not find config file located at: {args.config_path}")
    
    return args.config_path, args.eval


if __name__ == '__main__':
    cfg_path, eval_mode = get_cfg()

    cfgname = Path(cfg_path).stem
    logger = logging.getLogger('myLogger')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(f'/mnt/home/atanelus/Heap/localization_vae/{cfgname + ("_eval" if eval_mode else "")}.log'))
    logger.info("Initializing...")
    if eval_mode:
        eval(cfg_path, logger)
    else:
        train(cfg_path, logger)
