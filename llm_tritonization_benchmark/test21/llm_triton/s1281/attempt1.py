import torch
import triton
import triton.language as tl

@triton.jit
def expand_x_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to handle loop-carried dependency
    pid = tl.program_id(0)
    if pid != 0:
        return
    
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
def s1281_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Compute new values for a and b
    a_new = x_vals - 1.0
    b_new = x_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s1281_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded array for scalar x
    x_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar x with sequential processing
    grid = (1,)
    expand_x_kernel[grid](
        a, b, c, d, e, x_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Use expanded array in parallel computation
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s1281_kernel[grid](
        a, b, c, d, e, x_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )