import triton
import triton.language as tl
import torch

@triton.jit
def s112_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Read from copy (immutable) and b array
    a_vals = tl.load(a_copy_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute: a[i+1] = a[i] + b[i]
    result = a_vals + b_vals
    
    # Write to original array at offset i+1
    write_offsets = offsets + 1
    write_mask = write_offsets < (n_elements + 1)
    tl.store(a_ptr + write_offsets, result, mask=write_mask)

def s112_triton(a, b):
    n_elements = len(a) - 1  # Process indices 0 to LEN_1D-2
    
    # Create read-only copy to handle WAR dependencies
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