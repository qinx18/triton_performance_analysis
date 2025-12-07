import triton
import triton.language as tl
import torch

@triton.jit
def s131_kernel(a_ptr, a_copy_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load from read-only copy (offset by m=1) and b
    a_copy_vals = tl.load(a_copy_ptr + current_offsets + 1, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    
    # Compute result
    result = a_copy_vals + b_vals
    
    # Store to original array
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s131_triton(a, b, m):
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s131_kernel[(grid_size,)](
        a, a_copy, b,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )