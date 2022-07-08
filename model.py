import torch
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from utils.sparse_transformers import LearnedEncoding, SparseTransformerEncoder, SparseTransformerEncoderLayer


class VocalizationVAE(nn.Module):
    def __init__(self, 
        crop_size: int=4096,
        block_size: int=4,
        d_model: int=512,
        num_heads: int=8,
        latent_dim: int=16
    ):
        """ Constructs a variational autoencoder over the microphone traces
        of Mongolian gerbil vocalizations.
        Prameters:
            - crop_size: max length of the input sequence
            - block_size: number of audio samples per input token
            - d_model: inner dimension size of transformer layers
            - num_heads: Number of attention heads to use
            - latent_dim: Dimensionality of latent distribution
        """
        super(VocalizationVAE, self).__init__()
        self.crop_size = crop_size
        self.block_size = block_size
        self.max_seq = self.crop_size // self.block_size

        self.d_model = d_model
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.num_mics = 4

        self.in_encoding = LearnedEncoding(
            d_model=self.d_model, max_seq_len=self.max_seq + 1)
        self.out_encoding = LearnedEncoding(
            d_model=self.d_model, max_seq_len=self.max_seq)

        self.data_encoding = nn.Linear(self.block_size * self.num_mics, self.d_model)
        self.encoder = SparseTransformerEncoder(
            SparseTransformerEncoderLayer(
                self.d_model,
                self.num_heads,
                block_size=8,
                n_global=2,
                n_window=5,
                n_random=3,
                dim_feedforward=2048,
                dropout=0.1,
                checkpoint=False,
                batch_first=True
            ),
            5
        )

        # Encoder will produce two vectors of dim latent_dim, which encode
        # the mean and log(variance) of the posterior latent distribution
        self.post_mean = nn.Linear(self.d_model, self.latent_dim)
        self.post_logvar = nn.Linear(self.d_model, self.latent_dim)

        self.expansion = nn.Linear(self.latent_dim, self.max_seq * 2)

        # I really don't want this to use a kernel > 1
        self.conv_expansion = nn.Conv1d(
            in_channels=2,
            out_channels=self.d_model,
            kernel_size=1,
            padding='same'
        )

        self.decoder_blocks = nn.ModuleList([
            SparseTransformerEncoder(
                SparseTransformerEncoderLayer(
                    self.d_model,
                    self.num_heads,
                    block_size=8,
                    n_global=2,
                    n_window=5,
                    n_random=3,
                    dim_feedforward=2048,
                    dropout=0.1,
                    checkpoint=False,
                    batch_first=True
                ),
                5
            )
        ])

        self.to_seq = nn.Conv1d(
            in_channels=self.d_model,
            out_channels=self.block_size * self.num_mics,
            kernel_size=1
        )

    def _clip_gradients(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

    def encode(self, x):
        batched = x.dim() == 3

        # Initial shape: (batch, blocks, block_size * n_mics)
        embed_out = self.data_encoding(x)  # Outputs (batch, blocks, d_model)
        cls_token = torch.zeros((1, self.d_model), device=x.device)
        if batched:
            cls_token = cls_token.unsqueeze(0).expand(x.shape[0], -1, -1)
        transformer_in = torch.cat([cls_token, embed_out], dim=-2)
        transformer_in = self.in_encoding(
            transformer_in.unsqueeze(0) if not batched else transformer_in)
        transformer_out = self.encoder(transformer_in)[:, 0, :]
        mean, log_var = self.post_mean(
            transformer_out), self.post_logvar(transformer_out)
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
        # (1, latent_dim) -> (1, 1, max_seq_len * 2)
        seq = self.expansion(z).unsqueeze(1)
        # (1, 1, max_seq_len * 2) -> (1, 2, max_seq_len)
        seq = seq.reshape(seq.shape[0], 2, -1)
        # (1, 1, max_seq_len*2) -> (1, max_seq_len*2, 1)
        seq = self.conv_expansion(seq).transpose(1, 2)

        decode = self.out_encoding(seq)
        for decoder in self.decoder_blocks:
            decode = decoder(decode)
        decode = decode.transpose(1, 2)  # (1, seq, feat) -> (1, feat, seq)
        return self.to_seq(decode).transpose(-1, -2).reshape(z.shape[0], -1, self.num_mics)  # 4 microphone channels

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = VocalizationVAE.sample(mean, logvar)
        decoded = self.decode(z)
        return decoded, VocalizationVAE.elbo(mean, logvar)
