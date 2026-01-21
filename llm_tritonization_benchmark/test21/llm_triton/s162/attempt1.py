import triton
import triton.language as tl
import torch

@triton.jit
def s162_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, k, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < (n - 1)
    
    # Load from read-only copy for a[i + k]
    a_read_offsets = offsets + k
    a_read_mask = mask & (a_read_offsets < n)
    a_vals = tl.load(a_copy_ptr + a_read_offsets, mask=a_read_mask, other=0.0)
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Compute a[i] = a[i + k] + b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store to original array
    tl.store(a_ptr + offsets, result, mask=mask)

def s162_triton(a, b, c, k):
    n = a.shape[0]
    
    if k <= 0:
        return
    
    # Create read-only copy to handle WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n - 1, BLOCK_SIZE),)
    
    s162_kernel[grid](
        a, a_copy, b, c,
        k, n,
        BLOCK_SIZE=BLOCK_SIZE
    )