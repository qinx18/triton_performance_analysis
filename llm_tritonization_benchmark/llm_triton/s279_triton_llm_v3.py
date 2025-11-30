import torch
import triton
import triton.language as tl

@triton.jit
def s279_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Condition: a[i] > 0
    cond1 = a > 0.0
    
    # Path 1: a[i] <= 0
    # b[i] = -b[i] + d[i] * d[i]
    b_new = -b + d * d
    
    # Condition: b[i] <= a[i] (only for path 1)
    cond2 = b_new <= a
    
    # Path 1a: a[i] <= 0 and b[i] > a[i]
    # c[i] += d[i] * e[i]
    c_path1a = c + d * e
    
    # Path 1b: a[i] <= 0 and b[i] <= a[i]
    # c remains unchanged
    c_path1b = c
    
    # Combine path 1a and 1b
    c_path1 = tl.where(cond2, c_path1b, c_path1a)
    
    # Path 2: a[i] > 0
    # c[i] = -c[i] + e[i] * e[i]
    c_path2 = -c + e * e
    b_path2 = b  # b remains unchanged in this path
    
    # Select between paths based on condition 1
    c_final = tl.where(cond1, c_path2, c_path1)
    b_final = tl.where(cond1, b_path2, b_new)
    
    # L30: a[i] = b[i] + c[i] * d[i]
    a_final = b_final + c_final * d
    
    # Store results
    tl.store(a_ptr + offsets, a_final, mask=mask)
    tl.store(b_ptr + offsets, b_final, mask=mask)
    tl.store(c_ptr + offsets, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.numel()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )