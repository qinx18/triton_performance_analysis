import triton
import triton.language as tl
import torch

@triton.jit
def s152s_kernel(a_ptr, b_ptr, c_ptr, idx, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    b_val = tl.load(b_ptr + idx)
    c_val = tl.load(c_ptr + idx)
    result = b_val + c_val
    
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    result_vals = a_vals + result
    tl.store(a_ptr + current_offsets, result_vals, mask=mask)

@triton.jit
def s152_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    for i in range(n_elements):
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        b_val = d_val * e_val
        tl.store(b_ptr + i, b_val)
        
        # Call s152s functionality inline
        c_val = tl.load(c_ptr + i)
        add_val = b_val + c_val
        
        # Update all elements of a by adding add_val
        a_vals = tl.load(a_ptr + current_offsets, mask=mask)
        result_vals = a_vals + add_val
        tl.store(a_ptr + current_offsets, result_vals, mask=mask)

def s152_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s152_kernel[grid](
        a, b, c, d, e,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )