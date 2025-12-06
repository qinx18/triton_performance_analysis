import torch
import triton
import triton.language as tl

@triton.jit
def s261_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = 1 + pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load data
    a_vals = tl.load(a_ptr + current_offsets, mask=mask)
    b_vals = tl.load(b_ptr + current_offsets, mask=mask)
    c_prev_vals = tl.load(c_ptr + current_offsets - 1, mask=mask)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask)
    
    # Compute
    t1 = a_vals + b_vals
    new_a_vals = t1 + c_prev_vals
    t2 = c_vals * d_vals
    new_c_vals = t2
    
    # Store results
    tl.store(a_ptr + current_offsets, new_a_vals, mask=mask)
    tl.store(c_ptr + current_offsets, new_c_vals, mask=mask)

def s261_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    if n_elements <= 1:
        return a, c
    
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n_elements - 1, BLOCK_SIZE)
    
    s261_kernel[(num_blocks,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, c