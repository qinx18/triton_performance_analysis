import triton
import triton.language as tl
import torch

@triton.jit
def s162_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, n, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n
    
    # Load from read-only copy for a[i + k]
    a_read_offsets = idx + k
    a_read_mask = mask & (a_read_offsets < n + k)  # Ensure we don't read out of bounds
    a_vals = tl.load(a_copy_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + idx, mask=mask, other=0.0)
    
    # Compute a[i] = a[i + k] + b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store to original array
    tl.store(a_ptr + idx, result, mask=mask)

def s162_triton(a, b, c, k):
    n = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    if k > 0:
        # Create read-only copy for WAR race condition handling
        a_copy = a.clone()
        
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        
        s162_kernel[grid](
            a, a_copy, b, c, n, k,
            BLOCK_SIZE=BLOCK_SIZE
        )