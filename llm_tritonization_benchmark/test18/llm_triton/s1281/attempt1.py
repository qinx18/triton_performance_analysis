import torch
import triton
import triton.language as tl

@triton.jit
def expand_x_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    x_expanded_ptr,
    n_elements,
):
    # Single thread processes all elements sequentially to expand scalar x
    if tl.program_id(0) == 0:
        x_val = 0.0
        for i in range(n_elements):
            a_val = tl.load(a_ptr + i)
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            e_val = tl.load(e_ptr + i)
            
            x_val = b_val * c_val + a_val * d_val + e_val
            tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s1281_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    x_expanded_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + current_offsets, mask=mask)
    
    # Compute a[i] = x - 1.0
    a_vals = x_vals - 1.0
    
    # Store results
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    tl.store(b_ptr + current_offsets, x_vals, mask=mask)

def s1281_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create temporary array for expanded scalar x
    x_expanded = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar x to array
    grid = (1,)
    expand_x_kernel[grid](
        a, b, c, d, e,
        x_expanded,
        n_elements,
    )
    
    # Phase 2: Use expanded array in parallel computation
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s1281_kernel[grid](
        a, b, c, d, e,
        x_expanded,
        n_elements,
        BLOCK_SIZE,
    )