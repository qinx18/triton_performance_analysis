import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential processing with fixed output index
    output_idx = 0
    
    for block_start in range(0, n_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < n_elements
        
        # Load input arrays
        b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
        c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
        e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute results for both branches
        de_product = d_vals * e_vals
        result_pos = b_vals + de_product
        result_neg = c_vals + de_product
        
        # Select based on condition
        condition = b_vals > 0.0
        result = tl.where(condition, result_pos, result_neg)
        
        # Store results sequentially
        for i in range(BLOCK_SIZE):
            if block_start + i < n_elements:
                output_offset = output_idx + i
                if output_offset < n_elements:
                    scalar_result = tl.load(b_ptr + current_offsets, mask=(offsets == i), other=0.0)
                    scalar_d = tl.load(d_ptr + current_offsets, mask=(offsets == i), other=0.0)
                    scalar_e = tl.load(e_ptr + current_offsets, mask=(offsets == i), other=0.0)
                    scalar_c = tl.load(c_ptr + current_offsets, mask=(offsets == i), other=0.0)
                    
                    if tl.sum(tl.where(offsets == i, b_vals > 0.0, False)) > 0:
                        final_result = tl.sum(tl.where(offsets == i, result_pos, 0.0))
                    else:
                        final_result = tl.sum(tl.where(offsets == i, result_neg, 0.0))
                    
                    tl.store(a_ptr + output_offset, final_result)
        
        output_idx += BLOCK_SIZE

@triton.jit
def s124_kernel_simple(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Process elements sequentially in blocks
    for i in range(n_elements):
        # Load single elements
        b_val = tl.load(b_ptr + i)
        d_val = tl.load(d_ptr + i) 
        e_val = tl.load(e_ptr + i)
        
        if b_val > 0.0:
            result = b_val + d_val * e_val
        else:
            c_val = tl.load(c_ptr + i)
            result = c_val + d_val * e_val
            
        tl.store(a_ptr + i, result)

def s124_triton(a, b, c, d, e):
    n_elements = b.numel()
    BLOCK_SIZE = 256
    
    # Use simple sequential kernel since this has dependencies
    grid = (1,)
    s124_kernel_simple[grid](
        a, b, c, d, e, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a