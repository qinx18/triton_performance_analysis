import triton
import triton.language as tl
import torch

@triton.jit
def s424_kernel(
    flat_2d_array_ptr,
    a_ptr,
    xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
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
    LEN_1D = a.shape[0]
    n_elements = LEN_1D - 1
    vl = 63
    
    # Create xx array - it's flat_2d_array offset by vl
    # We need to ensure xx has enough space for the largest index (i+1 where i goes to LEN_1D-2)
    xx = torch.zeros_like(flat_2d_array)
    xx[:] = flat_2d_array[:]
    xx = xx[vl:]  # xx points to flat_2d_array + vl
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        flat_2d_array,
        a,
        xx,
        n_elements,
        BLOCK_SIZE,
    )
    
    return xx