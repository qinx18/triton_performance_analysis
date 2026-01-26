import torch
import triton
import triton.language as tl

@triton.jit
def s251_expand_s_kernel(s_expanded_ptr, b_ptr, c_ptr, d_ptr, n_elements):
    # Single thread processes all elements sequentially to expand scalar s
    s_val = 0.0
    for i in range(n_elements):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        d_val = tl.load(d_ptr + i)
        s_val = b_val + c_val * d_val
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s251_kernel(a_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    result = s_vals * s_vals
    
    tl.store(a_ptr + offsets, result, mask=mask)

def s251_triton(a, b, c, d):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s sequentially
    s251_expand_s_kernel[(1,)](
        s_expanded,
        b,
        c,
        d,
        n_elements
    )
    
    # Phase 2: Compute final results in parallel
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    s251_kernel[(num_blocks,)](
        a,
        s_expanded,
        n_elements,
        BLOCK_SIZE
    )