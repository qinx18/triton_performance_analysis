import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(
    a_ptr,
    a_copy_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Current indices for reading
    read_offsets = block_start + offsets
    
    # Indices for reading from a_copy (i + m where m = 1)
    read_a_offsets = read_offsets + 1
    
    # Mask for valid elements (i < LEN_1D - 1)
    mask = read_offsets < (n_elements - 1)
    
    # Load values
    a_vals = tl.load(a_copy_ptr + read_a_offsets, mask=mask)
    b_vals = tl.load(b_ptr + read_offsets, mask=mask)
    
    # Compute: a[i] = a[i + m] + b[i]
    result = a_vals + b_vals
    
    # Store result
    tl.store(a_ptr + read_offsets, result, mask=mask)

def s131_triton(a, b):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR race condition
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s131_kernel[grid](
        a,
        a_copy,
        b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )