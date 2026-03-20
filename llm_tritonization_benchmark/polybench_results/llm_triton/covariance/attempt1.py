import triton
import triton.language as tl
import torch

@triton.jit
def compute_mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j = tl.program_id(0)
    if j >= M:
        return
    
    # Compute mean for column j
    sum_val = 0.0
    for i_start in range(0, N, BLOCK_SIZE):
        i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
        i_mask = i_offsets < N
        
        data_offsets = i_offsets * M + j
        vals = tl.load(data_ptr + data_offsets, mask=i_mask, other=0.0)
        sum_val += tl.sum(vals)
    
    mean_val = sum_val / N
    tl.store(mean_ptr + j, mean_val)

@triton.jit
def subtract_mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    if i >= N:
        return
    
    for j_start in range(0, M, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        j_mask = j_offsets < M
        
        data_offsets = i * M + j_offsets
        mean_offsets = j_offsets
        
        data_vals = tl.load(data_ptr + data_offsets, mask=j_mask)
        mean_vals = tl.load(mean_ptr + mean_offsets, mask=j_mask)
        
        result_vals = data_vals - mean_vals
        tl.store(data_ptr + data_offsets, result_vals, mask=j_mask)

@triton.jit
def compute_covariance_kernel(data_ptr, cov_ptr, float_n, M: tl.constexpr, N: tl.constexpr, 
                            BM: tl.constexpr, BN: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    i_start = pid_m * BM
    j_start = pid_n * BN
    
    i_offsets = i_start + tl.arange(0, BM)
    j_offsets = j_start + tl.arange(0, BN)
    
    i_mask = i_offsets < M
    j_mask = j_offsets < M
    
    # Only compute upper triangular part
    valid_mask = i_mask[:, None] & j_mask[None, :] & (i_offsets[:, None] <= j_offsets[None, :])
    
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    
    for k_start in range(0, N, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < N
        
        # Load data[k, i] for all k in block, i in tile
        data_i_offsets = k_offsets[:, None] * M + i_offsets[None, :]
        data_i_mask = k_mask[:, None] & i_mask[None, :]
        data_i = tl.load(data_ptr + data_i_offsets, mask=data_i_mask, other=0.0)
        
        # Load data[k, j] for all k in block, j in tile
        data_j_offsets = k_offsets[:, None] * M + j_offsets[None, :]
        data_j_mask = k_mask[:, None] & j_mask[None, :]
        data_j = tl.load(data_ptr + data_j_offsets, mask=data_j_mask, other=0.0)
        
        # Compute dot product
        acc += tl.dot(data_i.trans(), data_j)
    
    # Normalize by (N-1)
    acc = acc / (float_n - 1.0)
    
    # Store upper triangular part
    cov_offsets = i_offsets[:, None] * M + j_offsets[None, :]
    tl.store(cov_ptr + cov_offsets, acc, mask=valid_mask)
    
    # Store lower triangular part (symmetric)
    symmetric_mask = i_mask[:, None] & j_mask[None, :] & (i_offsets[:, None] > j_offsets[None, :])
    cov_sym_offsets = j_offsets[None, :] * M + i_offsets[:, None]
    tl.store(cov_ptr + cov_sym_offsets, acc.trans(), mask=symmetric_mask)

def covariance_triton(cov, data, mean, float_n, M, N):
    # Phase 1: Compute column means
    BLOCK_SIZE = 64
    grid1 = (M,)
    compute_mean_kernel[grid1](data, mean, M, N, BLOCK_SIZE)
    
    # Phase 2: Subtract means from data
    grid2 = (N,)
    subtract_mean_kernel[grid2](data, mean, M, N, BLOCK_SIZE)
    
    # Phase 3: Compute covariance matrix
    BM = BN = 32
    BLOCK_K = 64
    grid3 = (triton.cdiv(M, BM), triton.cdiv(M, BN))
    compute_covariance_kernel[grid3](data, cov, float_n, M, N, BM, BN, BLOCK_K)