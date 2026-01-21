import torch
import triton
import triton.language as tl

@triton.jit
def s251_expand_s_kernel(b_ptr, c_ptr, d_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid != 0:
        return
    
    s_val = 0.0
    offsets = tl.arange(0, BLOCK_SIZE)
    
    block_start = 0
    while block_start < n_elements:
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        
        block_end = min(block_start + BLOCK_SIZE, n_elements)
        num_elements = block_end - block_start
        
        for i in tl.static_range(BLOCK_SIZE):
            if i < num_elements:
                s_val = b_vals[i] + c_vals[i] * d_vals[i]
                tl.store(s_expanded_ptr + (block_start + i), s_val)
        
        block_start += BLOCK_SIZE

@triton.jit
def s251_kernel(a_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    s_vals = tl.load(s_expanded_ptr + current_offsets, mask=mask, other=0.0)
    result = s_vals * s_vals
    
    tl.store(a_ptr + current_offsets, result, mask=mask)

def s251_triton(a, b, c, d):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    s_expanded = torch.zeros(n, device=a.device, dtype=a.dtype)
    
    grid_expand = (1,)
    s251_expand_s_kernel[grid_expand](
        b, c, d, s_expanded, n, BLOCK_SIZE=BLOCK_SIZE
    )
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s251_kernel[grid](
        a, s_expanded, n, BLOCK_SIZE=BLOCK_SIZE
    )