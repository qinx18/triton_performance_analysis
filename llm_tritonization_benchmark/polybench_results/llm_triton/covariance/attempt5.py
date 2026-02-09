import triton
import triton.language as tl
import torch

@triton.jit
def compute_mean_kernel(
    mean_ptr, data_ptr,
    float_n,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    m_idx = pid * BLOCK_SIZE_M + m_offsets
    m_mask = m_idx < M
    
    mean_acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_idx = n_start + n_offsets
        n_mask = n_idx < N
        
        data_offsets = n_idx[:, None] * M + m_idx[None, :]
        data_mask = n_mask[:, None] & m_mask[None, :]
        data_vals = tl.load(data_ptr + data_offsets, mask=data_mask, other=0.0)
        
        mean_acc += tl.sum(data_vals, axis=0)
    
    mean_vals = mean_acc / float_n
    tl.store(mean_ptr + m_idx, mean_vals, mask=m_mask)

@triton.jit
def subtract_mean_kernel(
    data_ptr, mean_ptr,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    
    n_idx = pid_n * BLOCK_SIZE_N + n_offsets
    m_idx = pid_m * BLOCK_SIZE_M + m_offsets
    
    n_mask = n_idx < N
    m_mask = m_idx < M
    
    mean_vals = tl.load(mean_ptr + m_idx, mask=m_mask, other=0.0)
    
    data_offsets = n_idx[:, None] * M + m_idx[None, :]
    data_mask = n_mask[:, None] & m_mask[None, :]
    
    data_vals = tl.load(data_ptr + data_offsets, mask=data_mask, other=0.0)
    data_vals = data_vals - mean_vals[None, :]
    tl.store(data_ptr + data_offsets, data_vals, mask=data_mask)

@triton.jit
def covariance_kernel(
    cov_ptr, data_ptr,
    float_n,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = tl.arange(0, BLOCK_SIZE_M)
    j_offsets = tl.arange(0, BLOCK_SIZE_M)
    
    i_idx = pid_i * BLOCK_SIZE_M + i_offsets
    j_idx = pid_j * BLOCK_SIZE_M + j_offsets
    
    i_mask = i_idx < M
    j_mask = j_idx < M
    
    upper_mask = i_idx[:, None] <= j_idx[None, :]
    valid_mask = i_mask[:, None] & j_mask[None, :] & upper_mask
    
    cov_acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_M], dtype=tl.float32)
    
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_idx = n_start + n_offsets
        n_mask = n_idx < N
        
        data_i_offsets = n_idx[:, None] * M + i_idx[None, :]
        data_j_offsets = n_idx[:, None] * M + j_idx[None, :]
        
        data_i_mask = n_mask[:, None] & i_mask[None, :]
        data_j_mask = n_mask[:, None] & j_mask[None, :]
        
        data_i_vals = tl.load(data_ptr + data_i_offsets, mask=data_i_mask, other=0.0)
        data_j_vals = tl.load(data_ptr + data_j_offsets, mask=data_j_mask, other=0.0)
        
        prod = data_i_vals[:, :, None] * data_j_vals[:, None, :]
        cov_acc += tl.sum(prod, axis=0)
    
    cov_vals = cov_acc / (float_n - 1.0)
    
    cov_offsets = i_idx[:, None] * M + j_idx[None, :]
    tl.store(cov_ptr + cov_offsets, cov_vals, mask=valid_mask)
    
    diag_mask = i_idx[:, None] == j_idx[None, :]
    off_diag_mask = valid_mask & (diag_mask == False)
    cov_ji_offsets = j_idx[:, None] * M + i_idx[None, :]
    tl.store(cov_ptr + cov_ji_offsets, cov_vals, mask=off_diag_mask)

def covariance_triton(cov, data, mean, float_n, M, N):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    
    grid_mean = (triton.cdiv(M, BLOCK_SIZE_M),)
    compute_mean_kernel[grid_mean](
        mean, data,
        float_n,
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    grid_subtract = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M))
    subtract_mean_kernel[grid_subtract](
        data, mean,
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    grid_cov = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(M, BLOCK_SIZE_M))
    covariance_kernel[grid_cov](
        cov, data,
        float_n,
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )