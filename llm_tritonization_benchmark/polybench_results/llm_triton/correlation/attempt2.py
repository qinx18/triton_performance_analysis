import triton
import triton.language as tl
import torch

@triton.jit
def compute_mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j = tl.program_id(0)
    if j >= M:
        return
    
    mean_val = 0.0
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_start in range(0, N, BLOCK_SIZE):
        i_indices = i_start + i_offsets
        mask = i_indices < N
        data_vals = tl.load(data_ptr + i_indices * M + j, mask=mask, other=0.0)
        mean_val += tl.sum(tl.where(mask, data_vals, 0.0))
    
    mean_val /= N
    tl.store(mean_ptr + j, mean_val)

@triton.jit
def compute_stddev_kernel(data_ptr, mean_ptr, stddev_ptr, eps, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j = tl.program_id(0)
    if j >= M:
        return
    
    mean_j = tl.load(mean_ptr + j)
    stddev_val = 0.0
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_start in range(0, N, BLOCK_SIZE):
        i_indices = i_start + i_offsets
        mask = i_indices < N
        data_vals = tl.load(data_ptr + i_indices * M + j, mask=mask, other=0.0)
        diff = data_vals - mean_j
        stddev_val += tl.sum(tl.where(mask, diff * diff, 0.0))
    
    stddev_val /= N
    stddev_val = tl.sqrt(stddev_val)
    stddev_val = tl.where(stddev_val <= eps, 1.0, stddev_val)
    tl.store(stddev_ptr + j, stddev_val)

@triton.jit
def normalize_data_kernel(data_ptr, mean_ptr, stddev_ptr, float_n, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    i = tl.program_id(0)
    if i >= N:
        return
    
    sqrt_float_n = tl.sqrt(float_n)
    j_offsets = tl.arange(0, BLOCK_SIZE_M)
    
    for j_start in range(0, M, BLOCK_SIZE_M):
        j_indices = j_start + j_offsets
        mask = j_indices < M
        
        data_vals = tl.load(data_ptr + i * M + j_indices, mask=mask)
        mean_vals = tl.load(mean_ptr + j_indices, mask=mask)
        stddev_vals = tl.load(stddev_ptr + j_indices, mask=mask)
        
        data_vals = (data_vals - mean_vals) / (sqrt_float_n * stddev_vals)
        tl.store(data_ptr + i * M + j_indices, data_vals, mask=mask)

@triton.jit
def compute_correlation_kernel(data_ptr, corr_ptr, M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    i_start = pid_m * BM
    j_start = pid_n * BN
    
    i_offsets = tl.arange(0, BM)
    j_offsets = tl.arange(0, BN)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    i_mask = i_indices < M
    j_mask = j_indices < M
    
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    k_offsets = tl.arange(0, BLOCK_K)
    
    for k_start in range(0, N, BLOCK_K):
        k_indices = k_start + k_offsets
        k_mask = k_indices < N
        
        data_i = tl.load(data_ptr + k_indices[:, None] * M + i_indices[None, :], 
                        mask=k_mask[:, None] & i_mask[None, :], other=0.0)
        data_j = tl.load(data_ptr + k_indices[:, None] * M + j_indices[None, :], 
                        mask=k_mask[:, None] & j_mask[None, :], other=0.0)
        
        acc += tl.dot(data_i.trans(), data_j)
    
    for bi in range(BM):
        for bj in range(BN):
            i_idx = i_start + bi
            j_idx = j_start + bj
            if i_idx < M and j_idx < M:
                if i_idx == j_idx:
                    tl.store(corr_ptr + i_idx * M + j_idx, 1.0)
                elif i_idx < j_idx:
                    val = acc[bi, bj]
                    tl.store(corr_ptr + i_idx * M + j_idx, val)
                    tl.store(corr_ptr + j_idx * M + i_idx, val)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_SIZE = 64
    BM = BN = 16
    BLOCK_K = 32
    
    # Phase 1: Compute mean
    grid1 = (M,)
    compute_mean_kernel[grid1](
        data, mean, M, N, BLOCK_SIZE
    )
    
    # Phase 2: Compute stddev
    grid2 = (M,)
    compute_stddev_kernel[grid2](
        data, mean, stddev, eps, M, N, BLOCK_SIZE
    )
    
    # Phase 3: Normalize data
    grid3 = (N,)
    normalize_data_kernel[grid3](
        data, mean, stddev, float_n, M, N, BLOCK_SIZE
    )
    
    # Phase 4: Compute correlation matrix
    grid4 = (triton.cdiv(M, BM), triton.cdiv(M, BN))
    compute_correlation_kernel[grid4](
        data, corr, M, N, BM, BN, BLOCK_K
    )