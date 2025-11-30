import triton
import triton.language as tl
import torch

@triton.jit
def s2710_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    len_1d: tl.constexpr,
    x,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < len_1d
    
    # Load arrays
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Main condition: if (a[i] > b[i])
    cond1 = a > b
    
    # If branch: a[i] > b[i]
    a_new = tl.where(cond1, a + b * d, a)
    
    # Nested if in true branch: if (LEN_1D > 10)
    if len_1d > 10:
        c_true_branch = tl.where(cond1, c + d * d, c)
    else:
        c_true_branch = tl.where(cond1, d * e + 1.0, c)
    
    # Else branch: a[i] <= b[i]
    b_new = tl.where(~cond1, a + e * e, b)
    
    # Nested if in false branch: if (x > 0.)
    cond2 = x > 0.0
    if cond2:
        c_false_branch = tl.where(~cond1, a + d * d, c_true_branch)
    else:
        c_false_branch = tl.where(~cond1, c + e * e, c_true_branch)
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(c_ptr + offsets, c_false_branch, mask=mask)

def s2710_triton(a, b, c, d, e, x):
    len_1d = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_1d, BLOCK_SIZE),)
    
    s2710_kernel[grid](
        a, b, c, d, e,
        len_1d,
        x,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b, c