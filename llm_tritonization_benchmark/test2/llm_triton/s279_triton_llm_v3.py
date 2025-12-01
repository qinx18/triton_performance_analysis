import triton
import triton.language as tl
import torch

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
    
    # Load all arrays
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Control flow logic using conditional operations
    # if (a[i] > 0.) goto L20; else continue to next line
    cond1 = a > 0.0
    
    # Path when a[i] <= 0:
    # b[i] = -b[i] + d[i] * d[i]
    b_new = -b + d * d
    
    # if (b[i] <= a[i]) goto L30; else continue
    cond2 = b_new <= a
    
    # c[i] += d[i] * e[i] (when a[i] <= 0 and b[i] > a[i])
    c_path1 = c + d * e
    
    # L20: c[i] = -c[i] + e[i] * e[i] (when a[i] > 0)
    c_path2 = -c + e * e
    
    # Select c based on conditions
    # If a[i] > 0, use c_path2
    # If a[i] <= 0 and b[i] <= a[i], use original c
    # If a[i] <= 0 and b[i] > a[i], use c_path1
    c_final = tl.where(cond1, c_path2, tl.where(cond2, c, c_path1))
    
    # Update b only when a[i] <= 0
    b_final = tl.where(cond1, b, b_new)
    
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