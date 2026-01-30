import torch
import triton
import triton.language as tl

@triton.jit
def expand_x_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Single thread processes all elements sequentially to expand scalar x
    if tl.program_id(0) != 0:
        return
    
    x_val = 0.0
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load arrays
        a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
        
        # Process each element in the block sequentially
        for j in range(BLOCK_SIZE):
            if i + j < n_elements:
                # Compute x value
                x_val = b_vals[j] * c_vals[j] + a_vals[j] * d_vals[j] + e_vals[j]
                # Store expanded x value
                tl.store(x_expanded_ptr + i + j, x_val)

@triton.jit
def s1281_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask, other=0.0)
    
    # Compute a[i] = x - 1.0 and b[i] = x
    a_new = x_vals - 1.0
    b_new = x_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s1281_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    
    # Create expanded x array
    x_expanded = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar x
    grid = (1,)
    expand_x_kernel[grid](
        a, b, c, d, e, x_expanded,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Parallel computation using expanded x
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s1281_kernel[grid](
        a, b, c, d, e, x_expanded,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )