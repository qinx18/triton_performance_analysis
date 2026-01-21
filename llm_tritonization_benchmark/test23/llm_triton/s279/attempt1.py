import triton
import triton.language as tl
import torch

@triton.jit
def s279_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load all arrays
    a_vals = tl.load(a_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # if (a[i] > 0.0) goto L20; else continue to next statement
    cond1 = a_vals > 0.0
    
    # b[i] = -b[i] + d[i] * d[i] (only when a[i] <= 0.0)
    new_b = tl.where(cond1, b_vals, -b_vals + d_vals * d_vals)
    
    # if (b[i] <= a[i]) goto L30; else continue (only when a[i] <= 0.0)
    cond2 = new_b <= a_vals
    cond2 = cond2 & (~cond1)  # Only apply when we didn't go to L20
    
    # c[i] += d[i] * e[i] (only when a[i] <= 0.0 AND b[i] > a[i])
    c_update1 = c_vals + d_vals * e_vals
    new_c = tl.where(cond1, c_vals, tl.where(cond2, c_vals, c_update1))
    
    # L20: c[i] = -c[i] + e[i] * e[i] (only when a[i] > 0.0)
    c_update2 = -c_vals + e_vals * e_vals
    new_c = tl.where(cond1, c_update2, new_c)
    
    # L30: a[i] = b[i] + c[i] * d[i] (always executed)
    new_a = new_b + new_c * d_vals
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(b_ptr + idx, new_b, mask=mask)
    tl.store(c_ptr + idx, new_c, mask=mask)

def s279_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s279_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )