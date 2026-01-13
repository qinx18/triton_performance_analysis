import torch
import triton
import triton.language as tl

@triton.jit
def s254_expand_x_kernel(
    b_ptr,
    x_expanded_ptr,
    n_elements,
):
    # Single thread processes all elements sequentially
    x_val = tl.load(b_ptr + n_elements - 1)  # x = b[LEN_1D-1]
    
    for i in range(n_elements):
        tl.store(x_expanded_ptr + i, x_val)
        x_val = tl.load(b_ptr + i)  # x = b[i] for next iteration

@triton.jit
def s254_kernel(
    a_ptr,
    b_ptr,
    x_expanded_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    result = (b_vals + x_vals) * 0.5
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s254_triton(a, b):
    n_elements = a.shape[0]
    
    # Create expanded x array
    x_expanded = torch.zeros_like(b)
    
    # Phase 1: Expand scalar x to array (sequential)
    s254_expand_x_kernel[(1,)](
        b,
        x_expanded,
        n_elements,
    )
    
    # Phase 2: Parallel computation
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s254_kernel[grid](
        a,
        b,
        x_expanded,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )