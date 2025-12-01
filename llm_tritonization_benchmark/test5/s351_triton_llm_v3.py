import triton
import triton.language as tl
import torch

@triton.jit
def s351_kernel(a_ptr, b_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for unroll_start in range(0, 5):
        current_offsets = block_start + offsets + unroll_start
        mask = current_offsets < n_elements
        
        b_vals = tl.load(b_ptr + current_offsets, mask=mask)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        
        result = a_vals + alpha * b_vals
        
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s351_triton(a, b, c):
    alpha = c[0].item()
    n_elements = a.shape[0]
    
    # Round up to nearest multiple of 5 for unrolling
    padded_n = ((n_elements + 4) // 5) * 5
    BLOCK_SIZE = 256
    grid = ((padded_n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s351_kernel[grid](
        a, b, alpha, n_elements, BLOCK_SIZE
    )
    
    return a