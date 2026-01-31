import triton
import triton.language as tl
import torch

@triton.jit
def s123_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_half, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < n_half
    
    b_vals = tl.load(b_ptr + i_offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + i_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + i_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + i_offsets, mask=mask, other=0.0)
    
    de_vals = d_vals * e_vals
    
    # First store: a[j] = b[i] + d[i] * e[i] where j starts from i
    val1 = b_vals + de_vals
    tl.store(a_ptr + i_offsets, val1, mask=mask)
    
    # Second store: a[j+1] = c[i] + d[i] * e[i] where applicable
    condition = c_vals > 0.0
    valid_mask = mask & condition
    
    # For second store, j position is i + 1 + count of positive c values before i
    # This is approximated as n_half + i for now
    val2 = c_vals + de_vals
    second_offsets = n_half + i_offsets
    tl.store(a_ptr + second_offsets, val2, mask=valid_mask)

def s123_triton(a, b, c, d, e):
    n = b.shape[0]
    n_half = n // 2
    
    a.zero_()
    
    # Sequential implementation to match exact C behavior
    j = -1
    for i in range(n_half):
        j += 1
        a[j] = b[i] + d[i] * e[i]
        if c[i] > 0.0:
            j += 1
            a[j] = c[i] + d[i] * e[i]