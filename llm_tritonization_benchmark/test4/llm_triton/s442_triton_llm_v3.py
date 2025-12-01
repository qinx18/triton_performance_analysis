import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = (block_start + offsets) < n_elements
    
    # Load data
    a_ptrs = a_ptr + block_start + offsets
    b_ptrs = b_ptr + block_start + offsets
    c_ptrs = c_ptr + block_start + offsets
    d_ptrs = d_ptr + block_start + offsets
    e_ptrs = e_ptr + block_start + offsets
    indx_ptrs = indx_ptr + block_start + offsets
    
    a_vals = tl.load(a_ptrs, mask=mask)
    b_vals = tl.load(b_ptrs, mask=mask)
    c_vals = tl.load(c_ptrs, mask=mask)
    d_vals = tl.load(d_ptrs, mask=mask)
    e_vals = tl.load(e_ptrs, mask=mask)
    indx_vals = tl.load(indx_ptrs, mask=mask)
    
    # Compute based on switch statement
    # case 1: a[i] += b[i] * b[i]
    case1_mask = (indx_vals == 1) & mask
    a_vals = tl.where(case1_mask, a_vals + b_vals * b_vals, a_vals)
    
    # case 2: a[i] += c[i] * c[i]
    case2_mask = (indx_vals == 2) & mask
    a_vals = tl.where(case2_mask, a_vals + c_vals * c_vals, a_vals)
    
    # case 3: a[i] += d[i] * d[i]
    case3_mask = (indx_vals == 3) & mask
    a_vals = tl.where(case3_mask, a_vals + d_vals * d_vals, a_vals)
    
    # case 4: a[i] += e[i] * e[i]
    case4_mask = (indx_vals == 4) & mask
    a_vals = tl.where(case4_mask, a_vals + e_vals * e_vals, a_vals)
    
    # Store result
    tl.store(a_ptrs, a_vals, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    s442_kernel[grid](
        a, b, c, d, e, indx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a