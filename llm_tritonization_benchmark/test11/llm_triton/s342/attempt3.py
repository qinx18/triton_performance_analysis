import torch
import triton
import triton.language as tl

@triton.jit
def s342_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    block_offsets = block_start + offsets
    mask = block_offsets < n_elements
    
    # Load values for this block
    a_vals = tl.load(a_ptr + block_offsets, mask=mask, other=0.0)
    
    # Count positive elements before this block
    j_base = 0
    for prev_block in range(0, block_start, BLOCK_SIZE):
        prev_offsets = prev_block + offsets
        prev_mask = prev_offsets < tl.minimum(block_start, n_elements)
        prev_vals = tl.load(a_ptr + prev_offsets, mask=prev_mask, other=0.0)
        j_base += tl.sum(tl.where(prev_vals > 0.0, 1, 0))
    
    # Process elements in this block sequentially
    for local_i in range(BLOCK_SIZE):
        if block_start + local_i >= n_elements:
            break
            
        global_i = block_start + local_i
        a_val = tl.load(a_ptr + global_i)
        
        if a_val > 0.0:
            b_val = tl.load(b_ptr + j_base)
            tl.store(a_ptr + global_i, b_val)
            j_base += 1

def s342_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s342_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a