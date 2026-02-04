import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(a_ptr, a_copy_ptr, b_ptr, c_ptr, d_ptr, n, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n
    
    # Load values
    a_vals = tl.load(a_copy_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Load a[i+1] for the second statement
    a_next_vals = tl.load(a_copy_ptr + idx + 1, mask=mask & (idx + 1 < (n + 1)))
    
    # First statement: a[i] *= c[i]
    new_a_vals = a_vals * c_vals
    
    # Second statement: b[i] += a[i + 1] * d[i]
    new_b_vals = b_vals + a_next_vals * d_vals
    
    # Store results
    tl.store(a_ptr + idx, new_a_vals, mask=mask)
    tl.store(b_ptr + idx, new_b_vals, mask=mask)

def s212_triton(a, b, c, d):
    n = a.shape[0] - 1  # Loop goes from 0 to LEN_1D-1
    
    # Create read-only copy to handle WAR dependency
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, a_copy, b, c, d, n, BLOCK_SIZE
    )