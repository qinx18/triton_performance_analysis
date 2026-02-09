import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Initialize
    r_0 = tl.load(r_ptr)
    alpha = -r_0
    tl.store(y_ptr, alpha)
    beta = 1.0
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential loop over k
    for k in range(1, N):
        # Update beta
        beta = (1.0 - alpha * alpha) * beta
        
        # Compute sum = sum(r[k-i-1]*y[i] for i in range(k))
        sum_val = 0.0
        for block_start in range(0, k, BLOCK_SIZE):
            i_offsets = block_start + offsets
            mask = i_offsets < k
            
            r_indices = k - i_offsets - 1
            y_indices = i_offsets
            
            r_vals = tl.load(r_ptr + r_indices, mask=mask, other=0.0)
            y_vals = tl.load(y_ptr + y_indices, mask=mask, other=0.0)
            
            products = tl.where(mask, r_vals * y_vals, 0.0)
            sum_val += tl.sum(products)
        
        # Update alpha
        r_k = tl.load(r_ptr + k)
        alpha = -(r_k + sum_val) / beta
        
        # Compute z[i] = y[i] + alpha*y[k-i-1] for i in range(k)
        for block_start in range(0, k, BLOCK_SIZE):
            i_offsets = block_start + offsets
            mask = i_offsets < k
            
            y_forward_indices = i_offsets
            y_backward_indices = k - i_offsets - 1
            
            y_forward = tl.load(y_ptr + y_forward_indices, mask=mask, other=0.0)
            y_backward = tl.load(y_ptr + y_backward_indices, mask=mask, other=0.0)
            
            z_vals = tl.where(mask, y_forward + alpha * y_backward, 0.0)
            tl.store(z_ptr + i_offsets, z_vals, mask=mask)
        
        # Copy z to y: y[i] = z[i] for i in range(k)
        for block_start in range(0, k, BLOCK_SIZE):
            i_offsets = block_start + offsets
            mask = i_offsets < k
            
            z_vals = tl.load(z_ptr + i_offsets, mask=mask, other=0.0)
            tl.store(y_ptr + i_offsets, z_vals, mask=mask)
        
        # Set y[k] = alpha
        tl.store(y_ptr + k, alpha)

def durbin_triton(r, y, z, N):
    BLOCK_SIZE = 32
    
    grid = (1,)
    durbin_kernel[grid](r, y, z, N, BLOCK_SIZE)