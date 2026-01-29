import triton
import triton.language as tl
import torch

@triton.jit
def s319_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, sum_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Load input arrays
    c_vals = tl.load(c_ptr + current_offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + current_offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute a[i] = c[i] + d[i]
    a_vals = c_vals + d_vals
    
    # Compute b[i] = c[i] + e[i]
    b_vals = c_vals + e_vals
    
    # Store results
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)
    tl.store(b_ptr + current_offsets, b_vals, mask=mask)
    
    # Compute sum for this block (sum += a[i] + b[i])
    block_sum_vals = a_vals + b_vals
    block_sum = tl.sum(block_sum_vals)
    
    # Store block sum (will be reduced later)
    pid = tl.program_id(0)
    tl.store(sum_ptr + pid, block_sum)

def s319_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create temporary array for partial sums
    partial_sums = torch.zeros(grid_size, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s319_kernel[(grid_size,)](
        a, b, c, d, e, partial_sums,
        n_elements, BLOCK_SIZE
    )
    
    # Final reduction on CPU/GPU
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()