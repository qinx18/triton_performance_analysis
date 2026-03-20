import triton
import triton.language as tl
import torch

@triton.jit
def compute_mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j = tl.program_id(0)
    if j >= M:
        return
    
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
def covariance_kernel(data_ptr, cov_ptr, float_n, M: tl.constexpr, N: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i = pid_i
    j = pid_j
    
    if i >= M or j < i or j >= M:
        return
    
    sum_val = 0.0
    for k in range(N):
        data_ki_offset = k * M + i
        data_kj_offset = k * M + j
        
        data_ki = tl.load(data_ptr + data_ki_offset)
        data_kj = tl.load(data_ptr + data_kj_offset)
        
        sum_val += data_ki * data_kj
    
    cov_val = sum_val / (float_n - 1.0)
    
    cov_ij_offset = i * M + j
    cov_ji_offset = j * M + i
    
    tl.store(cov_ptr + cov_ij_offset, cov_val)
    tl.store(cov_ptr + cov_ji_offset, cov_val)

def covariance_triton(cov, data, mean, float_n, M, N):
    BLOCK_SIZE = 64
    
    # Phase 1: Compute column means
    grid1 = (M,)
    compute_mean_kernel[grid1](data, mean, M, N, BLOCK_SIZE)
    
    # Phase 2: Subtract means from data
    grid2 = (N,)
    subtract_mean_kernel[grid2](data, mean, M, N, BLOCK_SIZE)
    
    # Phase 3: Compute covariance matrix
    grid3 = (M, M)
    covariance_kernel[grid3](data, cov, float_n, M, N)