import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get block start position
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices (1 to n-2 for the main loop)
    mask = (indices >= 1) & (indices < n - 1)
    
    # Load data with masking
    b_copy_vals = tl.load(b_copy_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # For sequential processing due to RAW dependency
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx >= 1 and idx < n - 1:
            # Read current values
            b_prev = tl.load(b_ptr + (idx - 1))
            b_next = tl.load(b_copy_ptr + (idx + 1))
            c_val = tl.load(c_ptr + idx)
            d_val = tl.load(d_ptr + idx)
            e_val = tl.load(e_ptr + idx)
            
            # Compute and store a[i]
            a_val = b_prev + c_val * d_val
            tl.store(a_ptr + idx, a_val)
            
            # Compute and store b[i]
            b_val = b_next - e_val * d_val
            tl.store(b_ptr + idx, b_val)

def s211_triton(a, b, c, d, e):
    n = a.shape[0]
    
    # Create read-only copy for WAR safety
    b_copy = b.clone()
    
    # Launch parameters
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    s211_kernel[grid](
        a, b, b_copy, c, d, e, n,
        BLOCK_SIZE=BLOCK_SIZE
    )