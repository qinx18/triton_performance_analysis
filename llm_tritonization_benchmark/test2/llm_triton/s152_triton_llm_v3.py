import triton
import triton.language as tl
import torch

@triton.jit
def s152s_kernel(a_ptr, b_ptr, c_ptr, idx, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    b_val = tl.load(b_ptr + idx)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    result = b_val * c_vals
    tl.store(a_ptr + offsets, result, mask=mask)

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    i = pid
    
    if i >= n_elements:
        return
    
    # b[i] = d[i] * e[i]
    d_val = tl.load(d_ptr + i)
    e_val = tl.load(e_ptr + i)
    b_val = d_val * e_val
    tl.store(b_ptr + i, b_val)
    
    # Call s152s equivalent inline
    b_i = tl.load(b_ptr + i)
    
    # Process all elements of array 'a' using the updated b[i]
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(num_blocks):
        block_start = block_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        c_vals = tl.load(c_ptr + offsets, mask=mask)
        result = b_i * c_vals
        tl.store(a_ptr + offsets, result, mask=mask)

def s152_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    
    grid = (n_elements,)
    
    s152_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )