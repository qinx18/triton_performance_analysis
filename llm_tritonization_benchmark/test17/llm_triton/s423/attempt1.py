import torch
import triton
import triton.language as tl

@triton.jit
def s423_kernel(flat_2d_array_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Indices for reading (i) and writing (i+1)
    read_offsets = block_start + offsets
    write_offsets = read_offsets + 1
    
    # Masks
    read_mask = read_offsets < n_elements
    write_mask = write_offsets < (n_elements + 1)
    
    # Load data
    xx_vals = tl.load(flat_2d_array_ptr + 64 + read_offsets, mask=read_mask, other=0.0)
    a_vals = tl.load(a_ptr + read_offsets, mask=read_mask, other=0.0)
    
    # Compute
    result = xx_vals + a_vals
    
    # Store result
    tl.store(flat_2d_array_ptr + write_offsets, result, mask=write_mask)

def s423_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array, a, n_elements, BLOCK_SIZE
    )
    
    return flat_2d_array