import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N):
    BLOCK_SIZE = 128
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize
    y_0 = tl.load(r_ptr)
    tl.store(y_ptr, -y_0)
    
    beta = 1.0
    alpha = -y_0
    
    # Sequential k loop
    for k in range(1, N):
        # Update beta
        beta = (1.0 - alpha * alpha) * beta
        
        # Compute sum with parallel reduction
        sum_val = 0.0
        
        for block_start in range(0, k, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < k
            
            # Load r[k-i-1] and y[i]
            r_indices = k - current_offsets - 1
            r_vals = tl.load(r_ptr + r_indices, mask=mask, other=0.0)
            y_vals = tl.load(y_ptr + current_offsets, mask=mask, other=0.0)
            
            # Accumulate sum
            products = r_vals * y_vals
            sum_val += tl.sum(tl.where(mask, products, 0.0))
        
        # Update alpha
        r_k = tl.load(r_ptr + k)
        alpha = -(r_k + sum_val) / beta
        
        # Update z and y arrays in parallel
        for block_start in range(0, k, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < k
            
            # z[i] = y[i] + alpha*y[k-i-1]
            y_i = tl.load(y_ptr + current_offsets, mask=mask, other=0.0)
            y_k_minus_i = tl.load(y_ptr + (k - current_offsets - 1), mask=mask, other=0.0)
            z_vals = y_i + alpha * y_k_minus_i
            tl.store(z_ptr + current_offsets, z_vals, mask=mask)
            
            # y[i] = z[i]
            tl.store(y_ptr + current_offsets, z_vals, mask=mask)
        
        # Set y[k] = alpha
        tl.store(y_ptr + k, alpha)

def durbin_triton(r, y, z, N):
    grid = (1,)
    durbin_kernel[grid](r, y, z, N)