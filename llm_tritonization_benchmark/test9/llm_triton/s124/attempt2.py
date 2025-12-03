import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Sequential processing - each program handles the entire array
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j = 0
    
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
        pos_vals = b_vals + de_product
        neg_vals = c_vals + de_product
        
        # Process each element in the block
        elements_in_block = tl.minimum(BLOCK_SIZE, n_elements - block_start)
        for i in range(BLOCK_SIZE):
            elem_mask = i < elements_in_block
            if elem_mask:
                # Select value based on condition
                b_val = tl.load(b_ptr + block_start + i)
                d_val = tl.load(d_ptr + block_start + i)
                e_val = tl.load(e_ptr + block_start + i)
                
                if b_val > 0.0:
                    val = b_val + d_val * e_val
                else:
                    c_val = tl.load(c_ptr + block_start + i)
                    val = c_val + d_val * e_val
                
                # Store to output array at index j
                tl.store(a_ptr + j, val)
                j += 1

def s124_triton(a, b, c, d, e):
    n_elements = b.shape[0]
    
    BLOCK_SIZE = 256
    grid = (1,)  # Single program to maintain sequential order
    
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a