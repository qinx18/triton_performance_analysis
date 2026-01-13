import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(a_ptr, flat_2d_array_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load indices for reading (i) and writing (i+1)
    read_offsets = block_start + offsets
    write_offsets = read_offsets + 1
    
    # Masks for bounds checking
    read_mask = read_offsets < n_elements
    write_mask = write_offsets < (n_elements + 1)
    
    # Load data
    a_vals = tl.load(a_ptr + read_offsets, mask=read_mask, other=0.0)
    xx_vals = tl.load(flat_2d_array_ptr + 64 + read_offsets, mask=read_mask, other=0.0)
    
    # Compute result
    result = xx_vals + a_vals
    
    # Store result
    tl.store(flat_2d_array_ptr + write_offsets, result, mask=write_mask)

def s423_triton(a, flat_2d_array):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        a, flat_2d_array, n_elements, BLOCK_SIZE
    )