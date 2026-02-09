import triton
import triton.language as tl
import torch

@triton.jit
def correlation_kernel(
    data_ptr, mean_ptr, stddev_ptr, corr_ptr,
    M, N, float_n, eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate which column we're processing
    j = pid
    
    if j >= M:
        return
    
    # Step 1: Calculate mean for column j
    mean_sum = 0.0
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_start in range(0, N, BLOCK_SIZE):
        i_indices = i_start + i_offsets
        i_mask = i_indices < N
        
        # Load data[i][j] values
        data_indices = i_indices * M + j
        data_vals = tl.load(data_ptr + data_indices, mask=i_mask, other=0.0)
        mean_sum += tl.sum(data_vals)
    
    mean_val = mean_sum / float_n
    tl.store(mean_ptr + j, mean_val)
    
    # Step 2: Calculate stddev for column j
    stddev_sum = 0.0
    
    for i_start in range(0, N, BLOCK_SIZE):
        i_indices = i_start + i_offsets
        i_mask = i_indices < N
        
        # Load data[i][j] values
        data_indices = i_indices * M + j
        data_vals = tl.load(data_ptr + data_indices, mask=i_mask, other=0.0)
        diff = data_vals - mean_val
        stddev_sum += tl.sum(diff * diff, axis=0)
    
    stddev_val = stddev_sum / float_n
    stddev_val = tl.sqrt(stddev_val)
    stddev_val = tl.where(stddev_val <= eps, 1.0, stddev_val)
    tl.store(stddev_ptr + j, stddev_val)

@triton.jit 
def normalize_kernel(
    data_ptr, mean_ptr, stddev_ptr,
    M, N, float_n,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    i = pid
    
    if i >= N:
        return
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    sqrt_float_n = tl.sqrt(float_n)
    
    for j_start in range(0, M, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = j_indices < M
        
        # Load data[i][j], mean[j], stddev[j]
        data_indices = i * M + j_indices
        data_vals = tl.load(data_ptr + data_indices, mask=j_mask)
        mean_vals = tl.load(mean_ptr + j_indices, mask=j_mask)
        stddev_vals = tl.load(stddev_ptr + j_indices, mask=j_mask)
        
        # Normalize: data[i][j] = (data[i][j] - mean[j]) / (sqrt(float_n) * stddev[j])
        normalized_vals = (data_vals - mean_vals) / (sqrt_float_n * stddev_vals)
        tl.store(data_ptr + data_indices, normalized_vals, mask=j_mask)

@triton.jit
def correlation_matrix_kernel(
    data_ptr, corr_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr
):
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    
    i = pid_i
    j = pid_j + i + 1  # Only compute upper triangle where j > i
    
    if i >= M - 1 or j >= M:
        return
    
    # Calculate correlation coefficient corr[i][j]
    corr_sum = 0.0
    k_offsets = tl.arange(0, BLOCK_SIZE)
    
    for k_start in range(0, N, BLOCK_SIZE):
        k_indices = k_start + k_offsets
        k_mask = k_indices < N
        
        # Load data[k][i] and data[k][j]
        data_i_indices = k_indices * M + i
        data_j_indices = k_indices * M + j
        
        data_i_vals = tl.load(data_ptr + data_i_indices, mask=k_mask, other=0.0)
        data_j_vals = tl.load(data_ptr + data_j_indices, mask=k_mask, other=0.0)
        
        corr_sum += tl.sum(data_i_vals * data_j_vals)
    
    # Store symmetric values
    corr_ij_idx = i * M + j
    corr_ji_idx = j * M + i
    tl.store(corr_ptr + corr_ij_idx, corr_sum)
    tl.store(corr_ptr + corr_ji_idx, corr_sum)

@triton.jit
def set_diagonal_kernel(corr_ptr, M):
    pid = tl.program_id(axis=0)
    i = pid
    
    if i >= M:
        return
    
    # Set diagonal elements to 1.0
    diag_idx = i * M + i
    tl.store(corr_ptr + diag_idx, 1.0)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_SIZE = 128
    
    # Step 1: Calculate mean and stddev for each column
    grid = (triton.cdiv(M, 1),)
    correlation_kernel[grid](
        data, mean, stddev, corr,
        M, N, float_n, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 2: Normalize data
    grid = (triton.cdiv(N, 1),)
    normalize_kernel[grid](
        data, mean, stddev,
        M, N, float_n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 3: Calculate correlation matrix (upper triangle)
    grid = (M - 1, triton.cdiv(M, 1))
    correlation_matrix_kernel[grid](
        data, corr,
        M, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 4: Set diagonal elements to 1.0
    grid = (triton.cdiv(M, 1),)
    set_diagonal_kernel[grid](corr, M)