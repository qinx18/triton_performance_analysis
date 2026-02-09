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
    # Get program ID for parallelization across M dimension
    pid_m = tl.program_id(0)
    
    # Step 1: Compute mean for each column (parallelized over M)
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    m_idx = pid_m * BLOCK_SIZE_M + m_offsets
    m_mask = m_idx < M
    
    # Initialize mean accumulator
    mean_acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Accumulate sum for each column
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_idx = n_start + n_offsets
        n_mask = n_idx < N
        
        # Load data[n_idx, m_idx]
        data_offsets = n_idx[:, None] * M + m_idx[None, :]
        data_mask = n_mask[:, None] & m_mask[None, :]
        data_vals = tl.load(data_ptr + data_offsets, mask=data_mask, other=0.0)
        
        # Sum along N dimension
        mean_acc += tl.sum(data_vals, axis=0)
    
    # Compute final mean
    mean_vals = mean_acc / float_n
    tl.store(mean_ptr + m_idx, mean_vals, mask=m_mask)

@triton.jit
def subtract_mean_kernel(
    data_ptr, mean_ptr,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # Get program IDs
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    # Compute offsets
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    
    n_idx = pid_n * BLOCK_SIZE_N + n_offsets
    m_idx = pid_m * BLOCK_SIZE_M + m_offsets
    
    n_mask = n_idx < N
    m_mask = m_idx < M
    
    # Load mean values
    mean_vals = tl.load(mean_ptr + m_idx, mask=m_mask, other=0.0)
    
    # Load data[n_idx, m_idx], subtract mean, and store back
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
    # Get program IDs for parallelizing over upper triangle
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Compute actual indices
    i_offsets = tl.arange(0, BLOCK_SIZE_M)
    j_offsets = tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    i_idx = pid_i * BLOCK_SIZE_M + i_offsets
    j_idx = pid_j * BLOCK_SIZE_M + j_offsets
    
    i_mask = i_idx < M
    j_mask = j_idx < M
    
    # Only compute upper triangle (i <= j)
    valid_mask = i_idx[:, None] <= j_idx[None, :]
    
    # Initialize covariance accumulator
    cov_acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_M], dtype=tl.float32)
    
    # Accumulate covariance
    for n_start in range(0, N, BLOCK_SIZE_N):
        n_idx = n_start + n_offsets
        n_mask = n_idx < N
        
        # Load data[n_idx, i_idx] and data[n_idx, j_idx]
        data_i_offsets = n_idx[:, None] * M + i_idx[None, :]
        data_j_offsets = n_idx[:, None] * M + j_idx[None, :]
        
        data_i_mask = n_mask[:, None] & i_mask[None, :]
        data_j_mask = n_mask[:, None] & j_mask[None, :]
        
        data_i_vals = tl.load(data_ptr + data_i_offsets, mask=data_i_mask, other=0.0)
        data_j_vals = tl.load(data_ptr + data_j_offsets, mask=data_j_mask, other=0.0)
        
        # Compute outer product and accumulate
        prod = data_i_vals[:, :, None] * data_j_vals[:, None, :]
        cov_acc += tl.sum(prod, axis=0)
    
    # Normalize by (N-1)
    cov_vals = cov_acc / (float_n - 1.0)
    
    # Store to both upper and lower triangle
    store_mask = i_mask[:, None] & j_mask[None, :] & valid_mask
    
    # Store upper triangle cov[i, j]
    cov_offsets_ij = i_idx[:, None] * M + j_idx[None, :]
    tl.store(cov_ptr + cov_offsets_ij, cov_vals, mask=store_mask)
    
    # Store lower triangle cov[j, i] (transpose)
    transpose_mask = i_mask[:, None] & j_mask[None, :] & (i_idx[:, None] < j_idx[None, :])
    cov_offsets_ji = j_idx[:, None] * M + i_idx[None, :]
    tl.store(cov_ptr + cov_offsets_ji, tl.trans(cov_vals), mask=transpose_mask)

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
    grid_cov = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(M, BLOCK_SIZE_M))
    covariance_matrix_kernel[grid_cov](
        cov, data,
        float_n,
        M, N,
        BLOCK_SIZE_M, BLOCK_SIZE_N
    )