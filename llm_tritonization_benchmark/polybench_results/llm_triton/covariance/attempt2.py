import triton
import triton.language as tl
import torch

@triton.jit
def covariance_kernel(data_ptr, mean_ptr, cov_ptr, float_n, M, N, data_stride_0, data_stride_1, cov_stride_0, cov_stride_1):
    # Step 1: Compute mean for each column j
    j = tl.program_id(0)
    if j < M:
        # Initialize mean[j] = 0
        mean_val = 0.0
        
        # Sum over all rows for column j
        for i in range(N):
            data_idx = i * data_stride_0 + j * data_stride_1
            data_val = tl.load(data_ptr + data_idx)
            mean_val += data_val
        
        # Divide by N
        mean_val /= float_n
        tl.store(mean_ptr + j, mean_val)

@triton.jit
def subtract_mean_kernel(data_ptr, mean_ptr, M, N, data_stride_0, data_stride_1):
    # Step 2: Subtract mean from data
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    if (i < N) & (j < M):
        data_idx = i * data_stride_0 + j * data_stride_1
        data_val = tl.load(data_ptr + data_idx)
        mean_val = tl.load(mean_ptr + j)
        data_val -= mean_val
        tl.store(data_ptr + data_idx, data_val)

@triton.jit
def covariance_matrix_kernel(data_ptr, cov_ptr, float_n, M, N, data_stride_0, data_stride_1, cov_stride_0, cov_stride_1):
    # Step 3: Compute covariance matrix
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    # Only compute upper triangular part (j >= i)
    if (i < M) & (j < M) & (j >= i):
        cov_val = 0.0
        
        # Sum over all N samples
        for k in range(N):
            data_ki_idx = k * data_stride_0 + i * data_stride_1
            data_kj_idx = k * data_stride_0 + j * data_stride_1
            data_ki = tl.load(data_ptr + data_ki_idx)
            data_kj = tl.load(data_ptr + data_kj_idx)
            cov_val += data_ki * data_kj
        
        # Divide by (N-1)
        cov_val /= (float_n - 1.0)
        
        # Store in both positions [i][j] and [j][i]
        cov_ij_idx = i * cov_stride_0 + j * cov_stride_1
        cov_ji_idx = j * cov_stride_0 + i * cov_stride_1
        tl.store(cov_ptr + cov_ij_idx, cov_val)
        tl.store(cov_ptr + cov_ji_idx, cov_val)

def covariance_triton(cov, data, mean, float_n, M, N):
    # Ensure tensors are contiguous
    data = data.contiguous()
    mean = mean.contiguous()
    cov = cov.contiguous()
    
    # Step 1: Compute means
    grid = (M,)
    covariance_kernel[grid](
        data, mean, cov, float_n, M, N,
        data.stride(0), data.stride(1),
        cov.stride(0), cov.stride(1)
    )
    
    # Step 2: Subtract means from data
    grid = (N, M)
    subtract_mean_kernel[grid](
        data, mean, M, N,
        data.stride(0), data.stride(1)
    )
    
    # Step 3: Compute covariance matrix
    grid = (M, M)
    covariance_matrix_kernel[grid](
        data, cov, float_n, M, N,
        data.stride(0), data.stride(1),
        cov.stride(0), cov.stride(1)
    )