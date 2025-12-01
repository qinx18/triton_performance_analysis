import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(flat_2d_array_ptr, xx_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Read indices for xx and a
    read_idx = block_start + offsets
    read_mask = read_idx < n_elements
    
    # Write indices for flat_2d_array (offset by +1)
    write_idx = read_idx + 1
    
    # Load data
    xx_vals = tl.load(xx_ptr + read_idx, mask=read_mask, other=0.0)
    a_vals = tl.load(a_ptr + read_idx, mask=read_mask, other=0.0)
    
    # Compute
    result = xx_vals + a_vals
    
    # Store result
    tl.store(flat_2d_array_ptr + write_idx, result, mask=read_mask)

def s423_triton(flat_2d_array, a):
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1
    
    # Set up xx pointer (flat_2d_array + 64)
    vl = 64
    xx = flat_2d_array[vl:]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s423_kernel[grid](
        flat_2d_array, xx, a,
        n_elements, BLOCK_SIZE
    )
    
    return flat_2d_array