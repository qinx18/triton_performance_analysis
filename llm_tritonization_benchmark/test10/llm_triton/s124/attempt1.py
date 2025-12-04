import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize output index
    out_idx = 0
    
    # Process in blocks
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input data
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute values for both branches
        de_product = d_vals * e_vals
        branch1_vals = b_vals + de_product  # when b[i] > 0
        branch2_vals = c_vals + de_product  # when b[i] <= 0
        
        # Select based on condition
        condition = b_vals > 0.0
        result_vals = tl.where(condition, branch1_vals, branch2_vals)
        
        # Store results sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                scalar_result = tl.load(result_vals.handle + i * result_vals.dtype.primitive_ty.size_in_bytes())
                tl.store(a_ptr + out_idx, scalar_result)
                out_idx += 1

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    BLOCK_SIZE = 256
    
    # Launch kernel with single block since we need sequential output
    grid = (1,)
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )