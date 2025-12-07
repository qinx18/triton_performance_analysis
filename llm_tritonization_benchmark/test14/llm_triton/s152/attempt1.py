import torch
import triton
import triton.language as tl

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, n_elements, BLOCK_SIZE):
        current_offsets = i + offsets
        mask = current_offsets < n_elements
        
        # Load d and e values
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        
        # Compute b[i] = d[i] * e[i]
        b_vals = d_vals * e_vals
        tl.store(b_ptr + current_offsets, b_vals, mask=mask)
        
        # Load a and c values for s152s computation
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask)
        
        # Compute a[i] += b[i] * c[i] (s152s functionality)
        a_vals = a_vals + b_vals * c_vals
        tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s152_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (1,)
    
    s152_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )