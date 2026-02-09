import triton
import triton.language as tl
import torch

@triton.jit
def covariance_kernel(
    cov_ptr, data_ptr, mean_ptr,
    float_n,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    m_idx = pid_m * BLOCK_SIZE_M + m_offsets
    m_mask = m_idx < M
    
    # Step 1: Compute mean for each column
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
def covariance_matrix_kernel(
    cov_ptr, data_ptr,
    float_n,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Map pid to (i, j) where i <= j
    total_pairs = (M * (M + 1)) // 2
    pairs_per_block = BLOCK_SIZE_M
    
    pair_start = pid * pairs_per_block
    pair_offsets = tl.arange(0, BLOCK_SIZE_M)
    pair_idx = pair_start + pair_offsets
    pair_mask = pair_idx < total_pairs
    
    # Convert linear pair index to (i, j)
    i_idx = tl.zeros([BLOCK_SIZE_M], dtype=tl.int32)
    j_idx = tl.zeros([BLOCK_SIZE_M], dtype=tl.int32)
    
    for p in range(BLOCK_SIZE_M):
        if pair_start + p < total_pairs:
            # Find i, j for upper triangular indexing
            linear_idx = pair_start + p
            i = 0
            while (i + 1) * (i + 2) // 2 <= linear_idx:
                i += 1
            j = linear_idx - (i * (i + 1) // 2)
            i_idx = tl.where(pair_offsets == p, i, i_idx)
            j_idx = tl.where(pair_offsets == p, j, j_idx)
    
    # Compute covariance for valid pairs
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    cov_acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_idx = n_start + n_offsets
        n_mask = n_idx < N
        
        data_i_offsets = n_idx[:, None] * M + i_idx[None, :]
        data_j_offsets = n_idx[:, None] * M + j_idx[None, :]
        
        data_i_mask = n_mask[:, None] & pair_mask[None, :] & (i_idx[None, :] < M)
        data_j_mask = n_mask[:, None] & pair_mask[None, :] & (j_idx[None, :] < M)
        
        data_i_vals = tl.load(data_ptr + data_i_offsets, mask=data_i_mask, other=0.0)
        data_j_vals = tl.load(data_ptr + data_j_offsets, mask=data_j_mask, other=0.0)
        
        prod = data_i_vals * data_j_vals
        cov_acc += tl.sum(prod, axis=0)
    
    cov_vals = cov_acc / (float_n - 1.0)
    
    # Store results
    valid_mask = pair_mask & (i_idx < M) & (j_idx < M)
    
    cov_ij_offsets = i_idx * M + j_idx
    tl.store(cov_ptr + cov_ij_offsets, cov_vals, mask=valid_mask)
    
    # Store transpose for off-diagonal elements
    off_diag_mask = valid_mask & (i_idx != j_idx)
    cov_ji_offsets = j_idx * M + i_idx
    tl.store(cov_ptr + cov_ji_offsets, cov_vals, mask=off_diag_mask)

def covariance_triton(cov, data, mean, float_n, M, N):
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    
    # Step 1: Compute mean
    grid_mean = (triton.cdiv(M, BLOCK_SIZE_M),)
    covariance_kernel[grid_mean](
        cov, data, mean,
        float_n,
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Step 2: Subtract mean from data
    grid_subtract = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M))
    subtract_mean_kernel[grid_subtract](
        data, mean,
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )
    
    # Step 3: Compute covariance matrix
    total_pairs = (M * (M + 1)) // 2
    grid_cov = (triton.cdiv(total_pairs, BLOCK_SIZE_M),)
    covariance_matrix_kernel[grid_cov](
        cov, data,
        float_n,
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )