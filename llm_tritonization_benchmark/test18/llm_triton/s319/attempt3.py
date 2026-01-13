import torch
import triton
import triton.language as tl

@triton.jit
def s319_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Compute a[i] = c[i] + d[i]
    a_vals = c_vals + d_vals
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Compute b[i] = c[i] + e[i]
    b_vals = c_vals + e_vals
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    
    # Sum both a_vals and b_vals for this block
    a_sum = tl.sum(tl.where(mask, a_vals, 0.0))
    b_sum = tl.sum(tl.where(mask, b_vals, 0.0))
    total_sum = a_sum + b_sum
    
    # Store block sum for reduction
    tl.atomic_add(sum_ptr, total_sum)

def s319_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Initialize sum tensor
    sum_tensor = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    s319_kernel[(grid_size,)](
        a, b, c, d, e, sum_tensor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return sum_tensor.item()