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
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load all arrays
    a = tl.load(a_ptr + idx, mask=mask)
    b = tl.load(b_ptr + idx, mask=mask)
    c = tl.load(c_ptr + idx, mask=mask)
    d = tl.load(d_ptr + idx, mask=mask)
    e = tl.load(e_ptr + idx, mask=mask)
    
    # Control flow logic
    cond1 = a > 0.0
    
    # Path 1: a[i] <= 0 - always update b first
    b_new = -b + d * d
    
    # Then check if b[i] <= a[i] for c update
    cond2 = b_new <= a
    c_path1 = tl.where(cond2, c, c + d * e)
    
    # Path 2: a[i] > 0 (L20) - b stays unchanged, c gets new value
    c_path2 = -c + e * e
    
    # Select final values based on first condition
    b_final = tl.where(cond1, b, b_new)
    c_final = tl.where(cond1, c_path2, c_path1)
    
    # L30: Final computation for a
    a_final = b_final + c_final * d
    
    # Store results
    tl.store(a_ptr + idx, a_final, mask=mask)
    tl.store(b_ptr + idx, b_final, mask=mask)
    tl.store(c_ptr + idx, c_final, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )