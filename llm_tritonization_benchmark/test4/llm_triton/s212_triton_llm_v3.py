import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n
        
        # Load data
        a_val = tl.load(a_ptr + current_offsets, mask=mask)
        b_val = tl.load(b_ptr + current_offsets, mask=mask)
        c_val = tl.load(c_ptr + current_offsets, mask=mask)
        d_val = tl.load(d_ptr + current_offsets, mask=mask)
        
        # Load a[i+1] for the dependency
        next_offsets = current_offsets + 1
        next_mask = next_offsets < (n + 1)
        a_next = tl.load(a_ptr + next_offsets, mask=next_mask)
        
        # First statement: a[i] *= c[i]
        a_new = a_val * c_val
        tl.store(a_ptr + current_offsets, a_new, mask=mask)
        
        # Second statement: b[i] += a[i + 1] * d[i]
        b_new = b_val + a_next * d_val
        tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s212_triton(a, b, c, d):
    n = a.shape[0] - 1
    BLOCK_SIZE = 256
    
    s212_kernel[(1,)](
        a, b, c, d, n,
        BLOCK_SIZE=BLOCK_SIZE
    )