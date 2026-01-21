import triton
import triton.language as tl
import torch

@triton.jit
def s442_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, indx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    b_vals = tl.load(b_ptr + indices, mask=mask)
    c_vals = tl.load(c_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    indx_vals = tl.load(indx_ptr + indices, mask=mask, other=0)
    
    # Apply conditional updates based on indx values
    # Case 1: if indx[i] == 1, then a[i] += b[i] * b[i]
    update1 = tl.where((indx_vals == 1) & mask, b_vals * b_vals, 0.0)
    
    # Case 2: if indx[i] == 2, then a[i] += c[i] * c[i]  
    update2 = tl.where((indx_vals == 2) & mask, c_vals * c_vals, 0.0)
    
    # Case 3: if indx[i] == 3, then a[i] += d[i] * d[i]
    update3 = tl.where((indx_vals == 3) & mask, d_vals * d_vals, 0.0)
    
    # Case 4: if indx[i] == 4, then a[i] += e[i] * e[i]
    update4 = tl.where((indx_vals == 4) & mask, e_vals * e_vals, 0.0)
    
    # Apply exactly one update based on switch condition
    result = a_vals + update1 + update2 + update3 + update4
    
    # Store result
    tl.store(a_ptr + indices, result, mask=mask)

def s442_triton(a, b, c, d, e, indx):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s442_kernel[grid](
        a, b, c, d, e, indx, 
        n_elements, 
        BLOCK_SIZE=BLOCK_SIZE
    )