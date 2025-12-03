import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(
    a_ptr,
    flat_2d_array_ptr,
    xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Read indices: 0 to n_elements-2
    read_indices = block_start + offsets
    read_mask = read_indices < n_elements - 1
    
    # Write indices: 1 to n_elements-1
    write_indices = read_indices + 1
    
    # Load data
    flat_vals = tl.load(flat_2d_array_ptr + read_indices, mask=read_mask, other=0.0)
    a_vals = tl.load(a_ptr + read_indices, mask=read_mask, other=0.0)
    
    # Compute
    result = flat_vals + a_vals
    
    # Store
    tl.store(xx_ptr + write_indices, result, mask=read_mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0]
    
    # Create xx array with vl offset (vl = 63)
    vl = 63
    xx_size = n_elements + vl
    xx = torch.zeros(xx_size, dtype=a.dtype, device=a.device)
    
    # xx pointer starts at offset vl
    xx_offset = xx[vl:]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s424_kernel[grid](
        a,
        flat_2d_array,
        xx_offset,
        n_elements,
        BLOCK_SIZE,
    )
    
    return xx