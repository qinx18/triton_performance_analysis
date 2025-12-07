import torch
import triton
import triton.language as tl

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Prologue: a[1] = b[0] + c[1] * d[1]
    if tl.program_id(0) == 0:
        c_val = tl.load(c_ptr + 1)
        d_val = tl.load(d_ptr + 1)
        b_val = tl.load(b_copy_ptr + 0)
        result = b_val + c_val * d_val
        tl.store(a_ptr + 1, result)
    
    # Main parallel loop: i from 1 to n_elements-3
    main_loop_size = n_elements - 3
    for block_start in range(0, main_loop_size, BLOCK_SIZE):
        current_offsets = block_start + offsets + 1  # i starts from 1
        mask = current_offsets < (main_loop_size + 1)
        
        # b[i] = b_copy[i+1] - e[i] * d[i]
        b_read_offsets = current_offsets + 1
        e_vals = tl.load(e_ptr + current_offsets, mask=mask)
        d_vals = tl.load(d_ptr + current_offsets, mask=mask)
        b_read_vals = tl.load(b_copy_ptr + b_read_offsets, mask=mask)
        b_results = b_read_vals - e_vals * d_vals
        tl.store(b_ptr + current_offsets, b_results, mask=mask)
        
        # a[i+1] = b[i] + c[i+1] * d[i+1]
        a_write_offsets = current_offsets + 1
        c_read_offsets = current_offsets + 1
        d_read_offsets = current_offsets + 1
        a_mask = a_write_offsets < n_elements
        
        c_vals = tl.load(c_ptr + c_read_offsets, mask=a_mask)
        d_vals_shifted = tl.load(d_ptr + d_read_offsets, mask=a_mask)
        a_results = b_results + c_vals * d_vals_shifted
        tl.store(a_ptr + a_write_offsets, a_results, mask=a_mask)
    
    # Epilogue: b[n_elements-2] = b_copy[n_elements-1] - e[n_elements-2] * d[n_elements-2]
    if tl.program_id(0) == 0:
        epilogue_idx = n_elements - 2
        if epilogue_idx >= 1:
            e_val = tl.load(e_ptr + epilogue_idx)
            d_val = tl.load(d_ptr + epilogue_idx)
            b_val = tl.load(b_copy_ptr + epilogue_idx + 1)
            result = b_val - e_val * d_val
            tl.store(b_ptr + epilogue_idx, result)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR safety
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )