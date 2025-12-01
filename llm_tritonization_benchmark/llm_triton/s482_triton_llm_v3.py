import triton
import triton.language as tl
import torch

@triton.jit
def s482_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load values
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute a[i] += b[i] * c[i]
        result = a_vals + b_vals * c_vals
        tl.store(a_ptr + current_offsets, result, mask=mask)
        
        # Check break condition: if c[i] > b[i] for any valid element
        break_condition = (c_vals > b_vals) & mask
        should_break = tl.sum(break_condition.to(tl.int32)) > 0
        
        if should_break:
            break

def s482_triton(a, b, c):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (1,)
    s482_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )