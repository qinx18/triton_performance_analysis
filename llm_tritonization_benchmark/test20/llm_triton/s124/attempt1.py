import triton
import triton.language as tl
import torch

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel needs to handle the sequential nature of the original loop
    # where j is incremented for each i, so we process elements sequentially
    pid = tl.program_id(0)
    
    # Process one block at a time sequentially
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Load input data
    mask = (block_start + offsets) < n_elements
    b_vals = tl.load(b_ptr + block_start + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + block_start + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + block_start + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + block_start + offsets, mask=mask, other=0.0)
    
    # Compute values based on condition
    de_product = d_vals * e_vals
    positive_mask = b_vals > 0.0
    result_vals = tl.where(positive_mask, b_vals + de_product, c_vals + de_product)
    
    # Store results - since j increments for every i, output indices match input indices
    output_offsets = block_start + offsets
    tl.store(a_ptr + output_offsets, result_vals, mask=mask)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Ensure output tensor is properly sized
    if a.shape[0] < n_elements:
        a = a[:n_elements]
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )