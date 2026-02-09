import triton
import triton.language as tl
import torch

@triton.jit
def correlation_kernel(data_ptr, mean_ptr, stddev_ptr, corr_ptr, 
                      eps, float_n, M, N, 
                      data_stride0, data_stride1,
                      corr_stride0, corr_stride1):
    
    # Compute means
    j_offsets = tl.arange(0, 128)
    i_offsets = tl.arange(0, 128)
    
    for j_block in range(0, M, 128):
        j_indices = j_block + j_offsets
        j_mask = j_indices < M
        
        means = tl.zeros((128,), dtype=tl.float32)
        
        for i_block in range(0, N, 128):
            i_indices = i_block + i_offsets
            i_mask = i_indices < N
            
            # Load data block for mean computation
            for j_idx in range(128):
                if j_block + j_idx < M:
                    sum_val = 0.0
                    for i_idx in range(128):
                        if i_block + i_idx < N:
                            data_idx = (i_block + i_idx) * data_stride0 + (j_block + j_idx) * data_stride1
                            val = tl.load(data_ptr + data_idx)
                            sum_val += val
                    means = tl.where(j_offsets == j_idx, means + sum_val, means)
        
        # Finalize means and store
        means = means / float_n
        tl.store(mean_ptr + j_indices, means, mask=j_mask)
    
    # Compute standard deviations
    for j_block in range(0, M, 128):
        j_indices = j_block + j_offsets
        j_mask = j_indices < M
        
        # Load means for this block
        means = tl.load(mean_ptr + j_indices, mask=j_mask)
        stddevs = tl.zeros((128,), dtype=tl.float32)
        
        for i_block in range(0, N, 128):
            i_indices = i_block + i_offsets
            i_mask = i_indices < N
            
            # Compute variance contributions
            for j_idx in range(128):
                if j_block + j_idx < M:
                    mean_val = tl.load(mean_ptr + j_block + j_idx)
                    sum_sq = 0.0
                    for i_idx in range(128):
                        if i_block + i_idx < N:
                            data_idx = (i_block + i_idx) * data_stride0 + (j_block + j_idx) * data_stride1
                            val = tl.load(data_ptr + data_idx)
                            diff = val - mean_val
                            sum_sq += diff * diff
                    stddevs = tl.where(j_offsets == j_idx, stddevs + sum_sq, stddevs)
        
        # Finalize stddevs
        stddevs = stddevs / float_n
        stddevs = tl.sqrt(stddevs)
        stddevs = tl.where(stddevs <= eps, 1.0, stddevs)
        tl.store(stddev_ptr + j_indices, stddevs, mask=j_mask)
    
    # Center and reduce data
    sqrt_float_n = tl.sqrt(float_n)
    for i_block in range(0, N, 64):
        i_indices = i_block + tl.arange(0, 64)
        i_mask = i_indices < N
        
        for j_block in range(0, M, 64):
            j_indices = j_block + tl.arange(0, 64)
            j_mask = j_indices < M
            
            # Load means and stddevs
            for i_idx in range(64):
                if i_block + i_idx < N:
                    for j_idx in range(64):
                        if j_block + j_idx < M:
                            data_idx = (i_block + i_idx) * data_stride0 + (j_block + j_idx) * data_stride1
                            val = tl.load(data_ptr + data_idx)
                            mean_val = tl.load(mean_ptr + j_block + j_idx)
                            stddev_val = tl.load(stddev_ptr + j_block + j_idx)
                            
                            centered = val - mean_val
                            normalized = centered / (sqrt_float_n * stddev_val)
                            tl.store(data_ptr + data_idx, normalized)
    
    # Compute correlation matrix
    for i in range(M - 1):
        # Set diagonal element
        diag_idx = i * corr_stride0 + i * corr_stride1
        tl.store(corr_ptr + diag_idx, 1.0)
        
        for j in range(i + 1, M):
            corr_val = 0.0
            
            # Compute correlation between columns i and j
            for k_block in range(0, N, 128):
                k_indices = k_block + tl.arange(0, 128)
                k_mask = k_indices < N
                
                for k_idx in range(128):
                    if k_block + k_idx < N:
                        data_i_idx = (k_block + k_idx) * data_stride0 + i * data_stride1
                        data_j_idx = (k_block + k_idx) * data_stride0 + j * data_stride1
                        val_i = tl.load(data_ptr + data_i_idx)
                        val_j = tl.load(data_ptr + data_j_idx)
                        corr_val += val_i * val_j
            
            # Store symmetric values
            corr_ij_idx = i * corr_stride0 + j * corr_stride1
            corr_ji_idx = j * corr_stride0 + i * corr_stride1
            tl.store(corr_ptr + corr_ij_idx, corr_val)
            tl.store(corr_ptr + corr_ji_idx, corr_val)
    
    # Set final diagonal element
    final_diag_idx = (M - 1) * corr_stride0 + (M - 1) * corr_stride1
    tl.store(corr_ptr + final_diag_idx, 1.0)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    grid = (1,)
    
    correlation_kernel[grid](
        data, mean, stddev, corr,
        eps, float_n, M, N,
        data.stride(0), data.stride(1),
        corr.stride(0), corr.stride(1)
    )