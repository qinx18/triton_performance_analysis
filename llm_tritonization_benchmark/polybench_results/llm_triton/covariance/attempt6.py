import triton
import triton.language as tl

@triton.jit
def mean_kernel(data_ptr, mean_ptr, float_n, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j = tl.program_id(0)
    
    if j >= M:
        return
    
    mean_val = 0.0
    for i_start in range(0, N, BLOCK_SIZE):
        i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
        mask = i_offsets < N
        data_vals = tl.load(data_ptr + i_offsets * M + j, mask=mask, other=0.0)
        mean_val += tl.sum(data_vals)
    
    mean_val = mean_val / float_n
    tl.store(mean_ptr + j, mean_val)

@triton.jit
def subtract_mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    for j_start in range(0, M, BLOCK_SIZE):
        j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
        mask = j_offsets < M
        
        data_vals = tl.load(data_ptr + i * M + j_offsets, mask=mask)
        mean_vals = tl.load(mean_ptr + j_offsets, mask=mask)
        result = data_vals - mean_vals
        tl.store(data_ptr + i * M + j_offsets, result, mask=mask)

@triton.jit
def covariance_kernel(data_ptr, cov_ptr, float_n, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)

    i = pid_i
    j = pid_j

    if (i >= M) or ((j < i) or (j >= M)):
        return

    cov_val = 0.0
    for k_start in range(0, N, BLOCK_SIZE):
        k_offsets = k_start + tl.arange(0, BLOCK_SIZE)
        mask = k_offsets < N
        
        data_i = tl.load(data_ptr + k_offsets * M + i, mask=mask, other=0.0)
        data_j = tl.load(data_ptr + k_offsets * M + j, mask=mask, other=0.0)
        cov_val += tl.sum(data_i * data_j)

    cov_val = cov_val / (float_n - 1.0)
    tl.store(cov_ptr + i * M + j, cov_val)
    tl.store(cov_ptr + j * M + i, cov_val)

def covariance_triton(cov, data, mean, float_n, M, N):
    BLOCK_SIZE = 64
    
    # Phase 1: Compute mean
    grid = (M,)
    mean_kernel[grid](data, mean, float_n, M, N, BLOCK_SIZE)
    
    # Phase 2: Subtract mean from data
    grid = (N,)
    subtract_mean_kernel[grid](data, mean, M, N, BLOCK_SIZE)
    
    # Phase 3: Compute covariance matrix
    grid = (M, M)
    covariance_kernel[grid](data, cov, float_n, M, N, BLOCK_SIZE)