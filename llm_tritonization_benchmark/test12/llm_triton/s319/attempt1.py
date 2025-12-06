import torch
import triton
import triton.language as tl

@triton.jit
def s319_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load input arrays
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Compute and store results
    a_vals = c_vals + d_vals
    b_vals = c_vals + e_vals
    
    tl.store(a_ptr + idx, a_vals, mask=mask)
    tl.store(b_ptr + idx, b_vals, mask=mask)

@triton.jit
def s319_sum_kernel(a_ptr, b_ptr, partial_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + idx, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + idx, mask=mask, other=0.0)
    
    # Sum within block
    block_sum = tl.sum(a_vals + b_vals)
    
    # Store partial sum (first thread in block)
    if tl.program_id(0) == 0:
        tl.atomic_add(partial_sums_ptr, block_sum)
    else:
        if offsets[0] == 0:
            tl.atomic_add(partial_sums_ptr, block_sum)

def s319_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # First kernel: compute a[i] = c[i] + d[i] and b[i] = c[i] + e[i]
    s319_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second kernel: compute sum of a[i] + b[i]
    partial_sum = torch.zeros(1, device=a.device, dtype=a.dtype)
    s319_sum_kernel[grid](
        a, b, partial_sum,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return partial_sum.item()