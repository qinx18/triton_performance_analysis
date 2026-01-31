import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute block start
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    element_offsets = block_start + offsets
    
    # Create mask for valid elements
    mask = element_offsets < n_elements
    
    # Load all elements for this block
    b_vals = tl.load(b_ptr + element_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + element_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + element_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + element_offsets, mask=mask, other=0.0)
    
    # Compute de_product once
    de_product = d_vals * e_vals
    
    # Compute results based on condition
    condition = b_vals > 0.0
    results = tl.where(condition, b_vals + de_product, c_vals + de_product)
    
    # Store results
    tl.store(a_ptr + element_offsets, results, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s124_kernel[(grid_size,)](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a