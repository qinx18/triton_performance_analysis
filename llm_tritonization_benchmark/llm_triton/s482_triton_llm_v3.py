import triton
import triton.language as tl
import torch

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Find first position where c[i] > b[i] (break condition)
    should_break = c_vals > b_vals
    break_mask = tl.cumsum(should_break.to(tl.int32), axis=0) == 0
    
    # Only update elements before the break condition
    update_mask = mask & break_mask
    
    # Compute a[i] += b[i] * c[i]
    result = a_vals + b_vals * c_vals
    
    # Store results only for valid positions
    tl.store(a_ptr + offsets, result, mask=update_mask)

def s482_triton(a, b, c):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    s482_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )