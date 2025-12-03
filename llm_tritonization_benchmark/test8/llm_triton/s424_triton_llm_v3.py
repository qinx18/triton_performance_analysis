import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < (n_elements - 1)
    
    # Load flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + block_start + offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute xx[i+1] = flat_2d_array[i] + a[i]
    result = flat_vals + a_vals
    
    # Store to xx[i+1] (which is flat_2d_array[i+1])
    output_offsets = block_start + offsets + 1
    output_mask = (block_start + offsets) < (n_elements - 1)
    tl.store(flat_2d_array_ptr + output_offsets, result, mask=output_mask)

def s424_triton(a, flat_2d_array):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    s424_kernel[grid](
        flat_2d_array, a, n_elements, BLOCK_SIZE
    )
    
    return flat_2d_array