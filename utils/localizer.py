from itertools import chain

import geotorch
import torch
from torch import nn
from torch.nn import functional as F

from .model import VocalizationVAE


class LocalizerVAE(VocalizationVAE):
    def __init__(self,
        crop_size: int=4096,
        block_size: int=4,
        d_model: int=512,
        num_heads: int=8,
        latent_dim: int=16,
        n_class_layers: int=1,
        feedforward_dim: int=256
    ):
        super(LocalizerVAE, self).__init__(crop_size, block_size, d_model, num_heads, latent_dim)

        if n_class_layers < 1: 
            raise ValueError('Classification head must have at least one layer.')

        self.classification_layers = nn.ModuleList()
        layer_dims = [latent_dim] + (n_class_layers - 1) * [feedforward_dim] + [2]
        
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            self.classification_layers.append(nn.Linear(in_dim, out_dim))
            geotorch.orthogonal(self.classification_layers[-1], "weight")
    
    def forward(self, audio, *, eval_mode=False):
        latent_mean, logvar = self.encode(audio)
        z = VocalizationVAE.sample(latent_mean, logvar)
        
        pred = z
        for layer in self.classification_layers[:-1]:
            pred = layer(pred)
            pred = F.relu(pred)
        pred = self.classification_layers[-1](pred)

        decoded = self.decode(z)
        if not eval_mode:
            return decoded, VocalizationVAE.elbo(latent_mean, logvar), pred
        else:
            var = torch.exp(logvar)
            if len(self.classification_layers) > 1:
                raise ValueError("Classification layer is non-linear")
            # (batch, latent_dim) -> (batch, 1, latent_dim)
            # linear weight has shape (in_channels, out_channels) = (latent_dim, 2)
            # Var(aX + bY) = a**2 Var(x) + b**2 Var**(y)
            batch_size = var.shape[0]
            var = torch.matmul(var.unsqueeze(1), self.classification_layers[0].weight.T ** 2).reshape(batch_size, 2)
            return pred, var
    
    def param_groups(self):
        return self.parameters()
        # params = chain(*[mod.parameters() for mod in self.classification_layers])
        # return [{'params': list(params)}]
