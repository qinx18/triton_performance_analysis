import torch
import triton
import triton.language as tl

@triton.jit
def s162_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Read from copy for a[i + k] and original arrays for b[i] and c[i]
        read_offsets = current_offsets + k
        read_mask = mask & (read_offsets < (n + 1))
        
        a_vals = tl.load(a_copy_ptr + read_offsets, mask=read_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] = a[i + k] + b[i] * c[i]
        result = a_vals + b_vals * c_vals
        
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s162_triton(a, b, c, k):
    if k <= 0:
        return
    
    n = a.shape[0] - 1  # LEN_1D - 1
    
    # Create read-only copy for WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s162_kernel[grid](
        a, a_copy, b, c, k, n, BLOCK_SIZE
    )