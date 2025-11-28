import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load flat_2d_array[i] and a[i]
    flat_vals = tl.load(flat_2d_array_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Compute xx[i+1] = flat_2d_array[i] + a[i]
    result = flat_vals + a_vals
    
    # Store to xx[i+1] (offset by 1)
    tl.store(xx_ptr + offsets + 1, result, mask=mask)

def s424_triton(flat_2d_array, a):
    n_elements = flat_2d_array.shape[0] - 1  # LEN_1D - 1
    
    # Create xx array (offset view of flat_2d_array + 63)
    vl = 63
    xx = torch.zeros_like(flat_2d_array)
    xx[vl:] = flat_2d_array[vl:]  # Initialize xx as offset view
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        flat_2d_array, a, xx, n_elements, BLOCK_SIZE
    )
    
    return xx