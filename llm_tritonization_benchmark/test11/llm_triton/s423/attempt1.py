import triton
import triton.language as tl
import torch

@triton.jit
def s423_kernel(
    flat_2d_array_ptr,
    a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Read offsets for xx (flat_2d_array + 64)
    read_offsets = block_start + offsets + 64
    read_mask = read_offsets < n_elements + 64 - 1
    
    # Write offsets for flat_2d_array (i+1)
    write_offsets = block_start + offsets + 1
    write_mask = write_offsets < n_elements
    
    # Load offsets for a
    load_offsets = block_start + offsets
    load_mask = load_offsets < n_elements - 1
    
    # Load xx[i] = flat_2d_array[i + 64]
    xx_vals = tl.load(flat_2d_array_ptr + read_offsets, mask=read_mask, other=0.0)
    
    # Load a[i]
    a_vals = tl.load(a_ptr + load_offsets, mask=load_mask, other=0.0)
    
    # Compute xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1]
    final_mask = write_mask & load_mask
    tl.store(flat_2d_array_ptr + write_offsets, result, mask=final_mask)

def s423_triton(a, flat_2d_array):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements - 1, BLOCK_SIZE)
    
    s423_kernel[(grid_size,)](
        flat_2d_array,
        a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return flat_2d_array