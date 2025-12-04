import torch
import triton
import triton.language as tl

@triton.jit
def s112_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load from read-only copy and b
    a_vals = tl.load(a_copy_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+1] (write to original array at offset +1)
    write_indices = indices + 1
    write_mask = mask & (write_indices < (n_elements + 1))
    tl.store(a_ptr + write_indices, result, mask=write_mask)

def s112_triton(a, b):
    n_elements = a.shape[0] - 1  # Process indices 0 to LEN_1D-2
    
    # Create read-only copy
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s112_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )