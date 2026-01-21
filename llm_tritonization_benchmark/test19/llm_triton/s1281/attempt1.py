import triton
import triton.language as tl
import torch

@triton.jit
def s1281_expand_x_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar x
    pid = tl.program_id(axis=0)
    if pid != 0:
        return
    
    x_val = 0.0  # Initial value of scalar x
    
    for i in range(n_elements):
        # Load values for computation
        a_val = tl.load(a_ptr + i)
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        # Compute new x value
        x_val = b_val * c_val + a_val * d_val + e_val
        
        # Store expanded x value
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s1281_compute_kernel(a_ptr, b_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Compute a[i] = x - 1.0 and b[i] = x
    a_vals = x_vals - 1.0
    b_vals = x_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s1281_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    
    # Create expanded array for scalar x
    x_expanded = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar x to array using sequential processing
    grid_expand = (1,)  # Single block for sequential processing
    s1281_expand_x_kernel[grid_expand](
        a, b, c, d, e, x_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Use expanded array in parallel computation
    grid_compute = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s1281_compute_kernel[grid_compute](
        a, b, x_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )