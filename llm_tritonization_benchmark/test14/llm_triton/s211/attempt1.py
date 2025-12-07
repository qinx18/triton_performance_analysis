import triton
import triton.language as tl
import torch

@triton.jit
def s211_kernel(a_ptr, b_ptr, b_copy_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Handle prologue: a[1] = b[0] + c[1] * d[1]
    if tl.program_id(0) == 0:
        b_0 = tl.load(b_copy_ptr)
        c_1 = tl.load(c_ptr + 1)
        d_1 = tl.load(d_ptr + 1)
        a_1 = b_0 + c_1 * d_1
        tl.store(a_ptr + 1, a_1)
    
    # Main parallel loop: process iterations i = 1 to n_elements-3
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + 1
    
    mask = i_offsets < (n_elements - 2)
    
    # Producer: b[i] = b_copy[i+1] - e[i] * d[i]
    b_vals = tl.load(b_copy_ptr + i_offsets + 1, mask=mask)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask)
    b_new = b_vals - e_vals * d_vals
    tl.store(b_ptr + i_offsets, b_new, mask=mask)
    
    # Consumer: a[i+1] = b[i] + c[i+1] * d[i+1]
    c_vals = tl.load(c_ptr + i_offsets + 1, mask=mask)
    d_vals_next = tl.load(d_ptr + i_offsets + 1, mask=mask)
    a_new = b_new + c_vals * d_vals_next
    tl.store(a_ptr + i_offsets + 1, a_new, mask=mask)

@triton.jit
def s211_epilogue_kernel(b_ptr, b_copy_ptr, e_ptr, d_ptr, n_elements):
    # Handle epilogue: b[n_elements-2] = b_copy[n_elements-1] - e[n_elements-2] * d[n_elements-2]
    if tl.program_id(0) == 0:
        idx = n_elements - 2
        b_val = tl.load(b_copy_ptr + n_elements - 1)
        e_val = tl.load(e_ptr + idx)
        d_val = tl.load(d_ptr + idx)
        b_new = b_val - e_val * d_val
        tl.store(b_ptr + idx, b_new)

def s211_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    
    # Create read-only copy for WAR safety
    b_copy = b.clone()
    
    BLOCK_SIZE = 256
    main_iterations = n_elements - 2
    grid = (triton.cdiv(main_iterations, BLOCK_SIZE),)
    
    # Launch main kernel
    s211_kernel[grid](
        a, b, b_copy, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Launch epilogue kernel
    s211_epilogue_kernel[(1,)](
        b, b_copy, e, d, n_elements
    )