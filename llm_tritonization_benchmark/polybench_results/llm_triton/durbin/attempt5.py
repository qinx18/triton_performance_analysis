import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N):
    # Initialize
    r_0 = tl.load(r_ptr + 0)
    tl.store(y_ptr + 0, -r_0)
    
    beta = 1.0
    alpha = -r_0
    
    # Sequential k loop
    for k in range(1, N):
        # Update beta
        beta = (1.0 - alpha * alpha) * beta
        
        # Compute sum sequentially
        sum_val = 0.0
        for i in range(k):
            r_val = tl.load(r_ptr + (k - i - 1))
            y_val = tl.load(y_ptr + i)
            sum_val += r_val * y_val
        
        # Update alpha
        r_k = tl.load(r_ptr + k)
        alpha = -(r_k + sum_val) / beta
        
        # Update z array first
        for i in range(k):
            y_i = tl.load(y_ptr + i)
            y_k_minus_i_minus_1 = tl.load(y_ptr + (k - i - 1))
            z_val = y_i + alpha * y_k_minus_i_minus_1
            tl.store(z_ptr + i, z_val)
        
        # Copy z to y
        for i in range(k):
            z_val = tl.load(z_ptr + i)
            tl.store(y_ptr + i, z_val)
        
        # Set y[k] = alpha
        tl.store(y_ptr + k, alpha)

def durbin_triton(r, y, z, N):
    grid = (1,)
    durbin_kernel[grid](r, y, z, N)