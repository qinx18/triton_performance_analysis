import torch
import triton
import triton.language as tl

@triton.jit
def s251_expand_s_kernel(s_expanded_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Single thread processes all elements sequentially
    if pid == 0:
        for i in range(n_elements):
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            d_val = tl.load(d_ptr + i)
            s_val = b_val + c_val * d_val
            tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s251_kernel(a_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask)
    result = s_vals * s_vals
    
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s251_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar s
    grid_expand = (1,)
    s251_expand_s_kernel[grid_expand](
        s_expanded, b, c, d, n_elements, BLOCK_SIZE
    )
    
    # Phase 2: Compute a[i] = s * s in parallel
    grid_compute = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s251_kernel[grid_compute](
        a, s_expanded, n_elements, BLOCK_SIZE
    )