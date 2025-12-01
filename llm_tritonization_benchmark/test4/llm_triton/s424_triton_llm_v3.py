import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(flat_2d_array_ptr, a_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Define offsets once at kernel start
    offsets = tl.arange(0, BLOCK_SIZE)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    # Load data
    flat_vals = tl.load(flat_2d_array_ptr + current_offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute result
    result = flat_vals + a_vals
    
    # Store to xx[i+1] - offset by 1
    store_offsets = current_offsets + 1
    store_mask = current_offsets < n_elements  # Use original mask since we're storing at i+1
    
    tl.store(xx_ptr + store_offsets, result, mask=store_mask)

def s424_triton(flat_2d_array, a):
    n_elements = len(a) - 1  # LEN_1D - 1
    
    # Create xx array (needs to be at least LEN_1D size for xx[i+1] access)
    xx = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s424_kernel[grid](
        flat_2d_array, a, xx, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return xx