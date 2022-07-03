from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def pad_mat(
    mat: torch.Tensor,
    block_size: int
):
    """ Pads a matrix or batch of matrices (in which the two final dimensions are considered the row
    and column dims of each matrix) such that both dimensions are enlarged to the nearest multiple of
    block_size greater than or equal to their current length.
    
    Params:
        - mat (torch.Tensor): The matrix or batch of matrices to pad
        - block_size (int): Number by which the new dimensions will be divisible
    """
    nearest_mult_d1 = mat.shape[-2] + block_size - (mat.shape[-2] % block_size) if (mat.shape[-2] % block_size != 0) else mat.shape[-2]
    nearest_mult_d2 = mat.shape[-1] + block_size - (mat.shape[-1] % block_size) if (mat.shape[-1] % block_size != 0) else mat.shape[-1]
    padded = torch.zeros((*mat.shape[:-2], nearest_mult_d1, nearest_mult_d2), dtype=mat.dtype, device=mat.device)
    padded[..., :mat.shape[-2], :mat.shape[-1]] = mat
    return padded


def dense_attn(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor, *,
    block_size: int=None,
    mask: torch.Tensor=None
):
    """ Computes dense attention, in which every sequence element may attend to all others, on provided
    query, key, and value sets. Optionally emulates a block-sparse masking protocol.

    Params:
        - q: Query set. Has shape (..., seq_len, d_model)
        - k: Key set. Has shape (..., seq_len, d_model)
        - v: Value set. Has shape (..., seq_len, d_model)
        - block_size: When emulating a block-sparse attention, this specifies the block size to use
        - mask: When emulating a block-sparse attention module, this mask is added to attention weights
        prior to normalization
    """
    orig_len = q.shape[-2]
    scale_factor = np.sqrt(q.shape[-1])
    
    if block_size is not None:
        q, k, v = pad_mat(q, block_size), pad_mat(k, block_size), pad_mat(v, block_size)
    
    scores = torch.matmul(q / scale_factor, k.transpose(-1, -2))
    if mask is not None:
        if orig_len < q.shape[-2]:
            # Simulate the masking of sequence padding done in the sparse implementation
            diff = q.shape[-2] - orig_len
            mask[-diff:, :] = -np.inf
            mask[:, -diff:] = -np.inf
        scores = scores + mask
    scaled_scores = torch.softmax(scores, dim=-1)
    return torch.matmul(scaled_scores, v)[:orig_len, :]


def sparse_attn(q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor, *,
                block_size: int,
                n_global: int,
                n_window: int,
                n_random: int,
                verbose: Optional[bool]=False):
    """ Computes sparse attention over the provided keys and queries according to BigBird:
    <https://arxiv.org/abs/2007.14062>.
    Figure 6 in Appendix D is particularly useful in deciphering the math here.
    Expected input shapes:
        - q: (..., seq_len, d_model), float, gpu
        - k: (..., seq_len, d_model), float, gpu
        - v: (..., seq_len, d_model), float, gpu
        block_size: scalar, int
        n_global: scalar, int
        n_window: scalar, odd int
        n_random: scalar, int
    """
    # I think this setup only makes sense for a square attention matrix, but I'm going to
    # separate these two values just in case
    # Saving for the sake of masking padded entries after matrix multiplication
    orig_q_len = q.shape[-2]
    orig_v_len = v.shape[-2]
    
    if n_global < 0 or n_window < 0 or n_random < 0:
        raise ValueError('Attention parameters n_global, n_window, and n_random should be positive integers')
    
    if n_window % 2 != 1:
        raise ValueError('Attention parameter n_window should be an odd integer')
    
    if orig_q_len != k.shape[-2]:
        raise ValueError('Key and Query sequences should have the same length.')
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError('Query, key, and Value sequence elements should contain the same number of features.')
    
    if orig_q_len % block_size != 0:
        q = pad_mat(q, block_size)
        k = pad_mat(k, block_size)
    if orig_v_len % block_size != 0:
        v = pad_mat(v, block_size)
    
    if verbose:
        print('pad_q', q.shape)
        print('pad_k', k.shape)
        print('v', v.shape)
    
    # Define some useful values
    n_blocks = q.shape[-2] // block_size
    d_model = q.shape[-1]
    
    if n_global + n_window + n_random > n_blocks:
        return dense_attn(q, k, v)
    
    q = q / np.sqrt(d_model)
    q_block = q.view(*q.shape[:-2], n_blocks, block_size, d_model)
    q_toprow = q[..., :n_global*block_size, :]  # For the tokens that attend to all tokens
    q_sparse = q_block[..., n_global:, :, :]

    k_t_toprow = k.transpose(-1, -2)  # unblocked keys for global rows
    k_block = k.view(*k.shape[:-2], n_blocks, block_size, d_model)
    # Blocked keys for the global columns)
    k_block_global = k_block[..., :n_global, :, :]
    # Blocked keys for the remaining sparse columns
    k_block_sparse = k_block[..., n_global:, :, :]
    
    # Computations for top row of blocks:
    # All keys interact with first g queries
    # q (..., n_global*block_size, d) x K^T (..., d, pad_seq_len) -> (..., n_global*block_size, pad_seq_len)
    top_row_global_scores = torch.matmul(q_toprow, k_t_toprow)
    if verbose:
        print('top_row_global_scores', top_row_global_scores.shape)
    
    
    # Start with the global columns
    global_col = k_block_global.view(*k.shape[:-2], 1, n_global * block_size, d_model)
    global_col = global_col.expand(*k.shape[:-2], n_blocks - n_global, -1, -1)
    
    # Gather the windowed columns
    # Roll out the windows
    win_radius = n_window // 2
    windowed = torch.cat([torch.roll(k_block, shift, dims=-3) for shift in range(-win_radius, win_radius+1)[::-1]], dim=-2)
    windowed = windowed[..., n_global:, :, :]  # Get rid of the global rows
    
    # Gather the random columns
    # Computed s.t. there is no intersection between the random blocks and the window/global blocks
    win_start = lambda row: row + n_global - win_radius  # index of the first key block in the sliding window
    win_end = lambda row: row + n_global + win_radius  # index of the last key block in the window
    # Valid indices from which random blocks may be selected given a row (relative to the end of the global rows)
    # Subtracting n_global accounts for the truncation of k_block_sparse to remove the first `n_global` rows (cols?)
    rand_valid_idx = lambda row: [a - n_global for a in range(n_global, n_blocks) if a < win_start(row) or a > win_end(row)]
    # Save these to use when gathering from v
    rand_sampled_cols = [np.random.choice(rand_valid_idx(row), size=n_random, replace=False) for row in range(n_global, n_blocks)]
    # Is there a way to directly index this that avoids building an N^2 mask array and drops the for loop?
    rand_cols = []
    for valid_idx in rand_sampled_cols:
        # This reshape just flattens the -3 and -2 dimensions into one
        new_shape = (*k_block_sparse.shape[:-3], n_random * block_size, d_model)
        gather_row = k_block_sparse[..., valid_idx, :, :].reshape(new_shape)
        rand_cols.append(gather_row)
    rand_cols = torch.stack(rand_cols, dim=-3)
    
    # Finally merge everything into the dense array
    k_t_dense = torch.cat([global_col, windowed, rand_cols], dim=-2).transpose(-1, -2)
    
    if verbose:
        print('k_global_col', global_col.shape)
        print('k_windowed', windowed.shape)
        print('rand_cols', rand_cols.shape)

        print('q_sparse', q_sparse.shape)
        print('k_t_dense', k_t_dense.shape)
        
    sparse_attn_scores = torch.matmul(q_sparse, k_t_dense)
    
    if verbose:
        print('sparse_attn_scores', sparse_attn_scores.shape)
    # I think I can directly softmax into the -1 (num_attended_blocks) dim after merging it with -3 (block_size)
    
    
    # Retrieve appropriate value indices:
    # Global columns
    v_global_col = v[..., :n_global*block_size, :].unsqueeze(-3)
    v_global_col = v_global_col.expand(*v_global_col.shape[:-3], n_blocks - n_global, -1, -1)
    
    # Windowed columns
    v_block = v.view(*v.shape[:-2], n_blocks, block_size, d_model)
    v_window = torch.cat([torch.roll(v_block, shift, dims=-3) for shift in range(-win_radius, win_radius+1)[::-1]], dim=-2)
    v_window = v_window[..., n_global:, :, :]
    
    # Random columns
    v_rand = []
    for valid_idx in rand_sampled_cols:
        # This reshape just flattens the -3 and -2 dimensions into one
        new_shape = (*v_block.shape[:-3], n_random * block_size, d_model)
        gather_row = v_block[..., valid_idx, :, :].reshape(new_shape)
        v_rand.append(gather_row)
    v_rand = torch.stack(v_rand, dim=-3)
    
    # Merge them all together into the set of values attended by each corresponding key
    # Should have shape (..., block_size * n_attended_blocks, d_model) ?
    v_complete = torch.cat([v_global_col, v_window, v_rand], dim=-2)
    # print('v_complete', v_complete)
    if verbose:
        print('v_complete', v_complete.shape)
        print('v_global_col', v_global_col.shape)
        print('v_window', v_window.shape)
        print('v_rand', v_rand.shape)
    
    
    # Mask illegal (duplicate) key-query pairs
    # This comes from the overlap of the windows with the global columns
    dup_mask = torch.zeros_like(sparse_attn_scores)
    for row_diff in range(win_radius):
        # Indexes into the first n windows of the first row, which intersect with the global columns
        dup_mask[..., row_diff, :, n_global*block_size:(n_global+win_radius-row_diff)*block_size] = -np.inf
        # Indexes into the last n windows of the last row, which wrap around to intersect with the global columns
        if n_random == 0:
            dup_mask[..., -row_diff, :, -(n_random + win_radius - row_diff)*block_size:] = -np.inf
        else:
            dup_mask[..., -row_diff, :, -(n_random + win_radius - row_diff)*block_size:-n_random*block_size] = -np.inf
    # Mask pad elements
    num_pad = q.shape[-2] - orig_q_len
    # Here I'm only preventing the corresponding query elements from attending to anything because I can't
    # think of an easy way to transpose the sparse matrix
    
    
    # ===================================================================
    # This is broken and produces nan gradients and I don't know why
    
    # if num_pad > 0:
    #    dup_mask[..., -1, -num_pad:, :] = -np.inf
    # ===================================================================
    
    masked_attn_scores = sparse_attn_scores + dup_mask
    scaled_attn_scores_sparse = torch.softmax(masked_attn_scores, dim=-1)
    if verbose:
        print('Sum of scaled sparse attention scores', scaled_attn_scores_sparse.sum(dim=-1))
    
    # Compute the attention scores for the global (dense) rows
    # Start by masking the padded indices:
    dense_mask = torch.zeros_like(top_row_global_scores)
    if num_pad > 0:
        dense_mask[..., :, -num_pad:] = -np.inf
    masked_dense_scores = top_row_global_scores + dense_mask
    scaled_dense_scores = torch.softmax(masked_dense_scores, dim=-1)
    dense_output = torch.matmul(scaled_dense_scores, v)
    
    
    pad_sparse_output = torch.matmul(scaled_attn_scores_sparse, v_complete)
    # Flatten out the n_blocks (-3) and block_size (-2) dimensions
    pad_sparse_output = pad_sparse_output.view(*pad_sparse_output.shape[:-3], -1, d_model)
    if num_pad > 0:
        sparse_output = pad_sparse_output[..., :-num_pad, :]
    else:
        sparse_output = pad_sparse_output
    if verbose:
        print('pad_sparse_output', pad_sparse_output.shape)
        print('sparse_output', sparse_output.shape)
        print('dense_output', dense_output.shape)
    
    return torch.cat([dense_output, sparse_output], dim=-2)
