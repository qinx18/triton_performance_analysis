import torch
import triton
import triton.language as tl

@triton.jit
def s281_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load b[i] and c[i]
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask)
    
    # Load a[LEN_1D-i-1] (reverse indexing)
    reverse_offsets = n_elements - 1 - (block_start + offsets)
    reverse_mask = mask & (reverse_offsets >= 0)
    a_reverse_vals = tl.load(a_ptr + reverse_offsets, mask=reverse_mask)
    
    # Compute x = a[LEN_1D-i-1] + b[i] * c[i]
    x_vals = a_reverse_vals + b_vals * c_vals
    
    # Store a[i] = x - 1.0
    tl.store(a_ptr + block_start + offsets, x_vals - 1.0, mask=mask)
    
    # Store b[i] = x
    tl.store(b_ptr + block_start + offsets, x_vals, mask=mask)

def s281_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create read-only copy of 'a' for the reverse access pattern
    a_copy = a.clone()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    s281_kernel[grid](
        a_copy, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Copy results back to original tensors
    a.copy_(a_copy)