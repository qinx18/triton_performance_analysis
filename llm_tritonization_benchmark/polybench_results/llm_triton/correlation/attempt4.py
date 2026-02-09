import triton
import triton.language as tl
import torch

@triton.jit
def mean_kernel(
    data_ptr, mean_ptr,
    M, N, float_n,
    BLOCK_SIZE: tl.constexpr
):
    j_start = tl.program_id(axis=0) * BLOCK_SIZE
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = j_start + j_offsets
    j_mask = j_indices < M
    
    mean_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(N):
        data_indices = i * M + j_indices
        data_vals = tl.load(data_ptr + data_indices, mask=j_mask, other=0.0)
        mean_vals += data_vals
    
    mean_vals = mean_vals / float_n
    tl.store(mean_ptr + j_indices, mean_vals, mask=j_mask)

@triton.jit
def stddev_kernel(
    data_ptr, mean_ptr, stddev_ptr,
    M, N, float_n, eps,
    BLOCK_SIZE: tl.constexpr
):
    j_start = tl.program_id(axis=0) * BLOCK_SIZE
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = j_start + j_offsets
    j_mask = j_indices < M
    
    mean_vals = tl.load(mean_ptr + j_indices, mask=j_mask, other=0.0)
    stddev_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(N):
        data_indices = i * M + j_indices
        data_vals = tl.load(data_ptr + data_indices, mask=j_mask, other=0.0)
        diff = data_vals - mean_vals
        stddev_vals += diff * diff
    
    stddev_vals = stddev_vals / float_n
    stddev_vals = tl.sqrt(stddev_vals)
    stddev_vals = tl.where(stddev_vals <= eps, 1.0, stddev_vals)
    tl.store(stddev_ptr + j_indices, stddev_vals, mask=j_mask)

@triton.jit
def normalize_kernel(
    data_ptr, mean_ptr, stddev_ptr,
    M, N, float_n,
    BLOCK_SIZE: tl.constexpr
):
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    
    i_start = pid_i * BLOCK_SIZE
    j_start = pid_j * BLOCK_SIZE
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    i_mask = i_indices < N
    j_mask = j_indices < M
    
    sqrt_n = tl.sqrt(float_n)
    
    for ii in range(BLOCK_SIZE):
        i_idx = i_start + ii
        i_valid = i_idx < N
        if i_valid:
            data_indices = i_idx * M + j_indices
            mean_vals = tl.load(mean_ptr + j_indices, mask=j_mask, other=0.0)
            stddev_vals = tl.load(stddev_ptr + j_indices, mask=j_mask, other=1.0)
            data_vals = tl.load(data_ptr + data_indices, mask=j_mask, other=0.0)
            
            normalized_vals = (data_vals - mean_vals) / (sqrt_n * stddev_vals)
            tl.store(data_ptr + data_indices, normalized_vals, mask=j_mask)

@triton.jit
def correlation_kernel(
    data_ptr, corr_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr
):
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    
    i_start = pid_i * BLOCK_SIZE
    j_start = pid_j * BLOCK_SIZE
    
    for ii in range(BLOCK_SIZE):
        i_idx = i_start + ii
        i_valid = i_idx < M
        
        if i_valid:
            diag_idx = i_idx * M + i_idx
            tl.store(corr_ptr + diag_idx, 1.0)
        
        for jj in range(BLOCK_SIZE):
            j_idx = j_start + jj
            valid_idx = i_valid & (j_idx < M) & (j_idx > i_idx)
            if valid_idx:
                corr_sum = 0.0
                for k in range(N):
                    data_i_idx = k * M + i_idx
                    data_j_idx = k * M + j_idx
                    data_i_val = tl.load(data_ptr + data_i_idx)
                    data_j_val = tl.load(data_ptr + data_j_idx)
                    corr_sum += data_i_val * data_j_val
                
                corr_ij_idx = i_idx * M + j_idx
                corr_ji_idx = j_idx * M + i_idx
                tl.store(corr_ptr + corr_ij_idx, corr_sum)
                tl.store(corr_ptr + corr_ji_idx, corr_sum)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_SIZE = 16
    
    # Step 1: Calculate means
    grid = (triton.cdiv(M, BLOCK_SIZE),)
    mean_kernel[grid](
        data, mean,
        M, N, float_n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 2: Calculate stddev
    stddev_kernel[grid](
        data, mean, stddev,
        M, N, float_n, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 3: Normalize data
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(M, BLOCK_SIZE))
    normalize_kernel[grid](
        data, mean, stddev,
        M, N, float_n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 4: Calculate correlation matrix
    grid = (triton.cdiv(M, BLOCK_SIZE), triton.cdiv(M, BLOCK_SIZE))
    correlation_kernel[grid](
        data, corr,
        M, N,
        BLOCK_SIZE=BLOCK_SIZE
    )