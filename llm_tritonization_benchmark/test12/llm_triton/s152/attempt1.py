import torch
import triton
import triton.language as tl

@triton.jit
def s152_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < n_elements:
            # Load single elements
            d_val = tl.load(d_ptr + idx)
            e_val = tl.load(e_ptr + idx)
            
            # Compute b[i] = d[i] * e[i]
            b_val = d_val * e_val
            tl.store(b_ptr + idx, b_val)
            
            # Load values for s152s computation
            a_val = tl.load(a_ptr + idx)
            c_val = tl.load(c_ptr + idx)
            
            # Inline s152s: a[i] += b[i] * c[i]
            a_new = a_val + b_val * c_val
            tl.store(a_ptr + idx, a_new)

def s152_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s152_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )