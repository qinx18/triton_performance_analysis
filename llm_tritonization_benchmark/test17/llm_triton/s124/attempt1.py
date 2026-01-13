import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize output index counter for this block
    output_idx = block_start
    
    # Process elements in this block
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load input data
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute results for both branches
    de_product = d_vals * e_vals
    result_positive = b_vals + de_product
    result_negative = c_vals + de_product
    
    # Select based on condition
    condition = b_vals > 0.0
    result = tl.where(condition, result_positive, result_negative)
    
    # Store results sequentially
    for i in range(BLOCK_SIZE):
        if block_start + i < n_elements:
            output_offset = output_idx + i
            if output_offset < n_elements:
                element_result = tl.load(result_positive.handle + i * result_positive.stride if condition.handle[i] else result_negative.handle + i * result_negative.stride)
                tl.store(a_ptr + output_offset, element_result)

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )