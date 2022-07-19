import importlib
from typing import Optional

import torch
from torch import nn

from .attn_func import sparse_attn


# Optional dependency: LocalAttention
local_attn_loader = importlib.find_loader('local_attention')
LocalAttention = local_attn_loader.LocalAttention if local_attn_loader is not None else None


class MultiheadSparseAttn(nn.Module):
    """ Implements multi-head attention with a sparse attention span.
    Implementation of BigBird: <https://arxiv.org/abs/2007.14062>.
    
    I thought about subclassing nn.MultiheadAttention here, but it uses concatenated
    weight matrices for the attention heads, and I'm not sure if that's compatible
    with my implementation of the attention function. To make up for this, most of
    the function signatures will be conserved.
    """
    def __init__(self,
        embed_dim: int,
        num_heads: int,
        block_size: int,
        n_global: int,
        n_window: int,
        n_random: int,
        bias: bool=True,
        batch_first: bool=False,
        device: Optional[torch.device]=None,
        dtype: Optional[torch.dtype]=None
    ):
        super(MultiheadSparseAttn, self).__init__()

        param_args = {'device': device, 'dtype': dtype}
        self.d_model = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.use_bias = bias
        self.rand_pattern = None

        self.attn_block_size = block_size
        self.attn_n_global = n_global
        self.attn_n_window = n_window
        self.attn_n_random = n_random

        if embed_dim % num_heads != 0:
            raise ValueError('Attention head count must divide model embedding dimension.')
        self.d_head = embed_dim // num_heads

        self.q_proj_weight = nn.Parameter(torch.empty((num_heads, embed_dim, self.d_head), **param_args))
        self.k_proj_weight = nn.Parameter(torch.empty((num_heads, embed_dim, self.d_head), **param_args))
        self.v_proj_weight = nn.Parameter(torch.empty((num_heads, embed_dim, self.d_head), **param_args))
        
        self.register_parameter('q_proj_weight', self.q_proj_weight)
        self.register_parameter('k_proj_weight', self.k_proj_weight)
        self.register_parameter('w_proj_weight', self.v_proj_weight)

        if bias:
            self.q_proj_bias = nn.Parameter(torch.empty((num_heads, self.d_head), **param_args))
            self.k_proj_bias = nn.Parameter(torch.empty((num_heads, self.d_head), **param_args))
            self.v_proj_bias = nn.Parameter(torch.empty((num_heads, self.d_head), **param_args))
            self.register_parameter('q_proj_bias', self.q_proj_bias)
            self.register_parameter('k_proj_bias', self.k_proj_bias)
            self.register_parameter('v_proj_bias', self.v_proj_bias)
        
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias, **param_args)
        self._init_parameters()
    
    def _init_parameters(self):
        """ Initializes projection weights
        """
        nn.init.xavier_uniform_(self.q_proj_weight)
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        if self.use_bias:
            nn.init.constant_(self.q_proj_bias, 0)
            nn.init.constant_(self.k_proj_bias, 0)
            nn.init.constant_(self.v_proj_bias, 0)
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        is_batched = query.dim() == 3  # batch dim + sequence dim + feature dim
        if not self.batch_first and is_batched:
            # Move the batch dim to 0, which my implementation expects
            query, key, value = (a.transpose(0, 1) for a in (query, key, value))
        
        # Reshape for matmul broadcasting:
        # (batch, seq_len, embed) -> (batch, 1, seq_len, embed)
        query, key, value = (a.unsqueeze(-3) for a in (query, key, value))
        Q = torch.matmul(query, self.q_proj_weight)  # Has shape (batch, n_head, seq_len, d_head)
        K = torch.matmul(key, self.k_proj_weight)
        V = torch.matmul(value, self.v_proj_weight)
        if self.use_bias:
            if is_batched:
                Q_b = self.q_proj_bias[None, :, None, :]
                K_b = self.k_proj_bias[None, :, None, :]
                V_b = self.v_proj_bias[None, :, None, :]
            else:
                Q_b = self.q_proj_bias[:, None, :]
                K_b = self.k_proj_bias[:, None, :]
                V_b = self.v_proj_bias[:, None, :]
            Q = Q + Q_b
            K = K + K_b
            V = V + V_b
        

        # Should have shape (batch, n_head, seq_len, d_head)
        attn_unchecked_output = sparse_attn(
            Q, K, V,
            block_size = self.attn_block_size,
            n_global = self.attn_n_global,
            n_window = self.attn_n_window,
            n_random = self.attn_n_random,
            rand_idx = self.rand_pattern,
            return_rand_idx=self.rand_pattern is None
        )
        # If sparse_attn elects to use the dense attention function, rand_pattern will not be returned
        if len(attn_unchecked_output) == 2:
            if self.rand_pattern is None:
                del self.rand_pattern
                self.register_buffer('rand_pattern', attn_unchecked_output[1])
            attn_out, _ = attn_unchecked_output
        else:
            attn_out = attn_unchecked_output

        # Flatten for projection
        # Involves transposing and reshaping to (batch * seq_len, n_head * d_head)

        seq_len = attn_out.shape[-2]
        flat = attn_out.transpose(-2, -3)
        flat = flat.reshape(-1, flat.shape[-1] * flat.shape[-2])
        attn_out = self.out_proj(flat)

        # Reintroduce batch dim
        if is_batched:
            attn_out = attn_out.view(-1, seq_len, self.d_model)
            if not self.batch_first:
                attn_out = attn_out.transpose(0, 1)
        
        return attn_out


class LocalAttnLayer(nn.Module):
    """ Based open the local attention function located here:
    <https://github.com/lucidrains/local-attention>
    """
    def __init__(self, d_model, n_heads, causal):
        super().__init__()

        if LocalAttention is None:
            raise ModuleNotFoundError("The local attention module must be installed to instantiate the LocalAttnLayer. Consider running `pip install local_attention`.")
        self.n_heads = n_heads
        self.d_model = d_model
        self.self_attn = LocalAttention(
            dim=d_model//n_heads,
            window_size=128,
            causal=causal,
            look_backward=1,
            dropout=0.1
        )
        proj_shape = (n_heads, d_model, d_model//n_heads)
        bias_shape = (n_heads, 1, d_model//n_heads)
        q_proj, k_proj, v_proj = torch.empty(proj_shape), torch.empty(proj_shape), torch.empty(proj_shape)
        nn.init.xavier_uniform_(q_proj)
        nn.init.xavier_uniform_(k_proj)
        nn.init.xavier_uniform_(v_proj)
        q_bias, k_bias, v_bias = torch.empty(bias_shape), torch.empty(bias_shape), torch.empty(bias_shape)
        nn.init.uniform_(q_bias, -0.1, 0.1)
        nn.init.uniform_(k_bias, -0.1, 0.1)
        nn.init.uniform_(v_bias, -0.1, 0.1)
        self.q_proj = nn.Parameter(q_proj, requires_grad=True)
        self.k_proj = nn.Parameter(k_proj, requires_grad=True)
        self.v_proj = nn.Parameter(v_proj, requires_grad=True)
        self.q_bias = nn.Parameter(q_bias, requires_grad=True)
        self.k_bias = nn.Parameter(k_bias, requires_grad=True)
        self.v_bias = nn.Parameter(v_bias, requires_grad=True)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        # Using default internal feedforward dim of 2048 and dropout of 10%
        self.ff1 = nn.Linear(d_model, 2048)
        self.ff2 = nn.Linear(2048, d_model)

    def _ff_block(self, x):
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
        return self.dropout2(self.ff2(self.dropout1(F.relu(self.ff1(x)))))

    def forward(self, x, *args, **kwargs):
        batch_size = x.shape[0]
        # (batch, seq, feat) -> (batch, head, seq, feat)
        x = x.unsqueeze(1)
        new_shape = (x.shape[0] * self.n_heads, x.shape[2], self.d_model // self.n_heads)
        Q = (torch.matmul(x, self.q_proj) + self.q_bias[None, :]).view(new_shape)
        K = (torch.matmul(x, self.k_proj) + self.k_bias[None, :]).view(new_shape)
        V = (torch.matmul(x, self.v_proj) + self.v_bias[None, :]).view(new_shape)
        
        self_attn_res = self.self_attn(Q, K, V)
        # Combine the results of all attn heads:
        # (batch * n_head, seq_len, dmodel//n_head)
        # -> (batch, n_head, seq_len, d_model//n_head)
        # -> (batch, seq_len, n_head, d_model//n_head)
        # -> (batch, seq_len, d_model)
        self_attn_res = self_attn_res.reshape(batch_size, self.n_heads, -1, self.d_model//self.n_heads)
        self_attn_res = self_attn_res.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        # (batch, 1, seq, feat) -> (batch, seq, feat)
        x = x.squeeze(1)
        x = self.norm1(x + self_attn_res)
        x = self.norm2(x + self._ff_block(x))
        return x
