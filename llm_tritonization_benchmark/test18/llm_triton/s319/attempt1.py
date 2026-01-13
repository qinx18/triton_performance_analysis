import triton
import triton.language as tl
import torch

@triton.jit
def s319_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    N,
    sum_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate offset range for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < N
    
    # Load arrays
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    e_vals = tl.load(e_ptr + offsets, mask=mask, other=0.0)
    
    # Compute a[i] = c[i] + d[i]
    a_vals = c_vals + d_vals
    
    # Compute b[i] = c[i] + e[i]  
    b_vals = c_vals + e_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    tl.store(b_ptr + offsets, b_vals, mask=mask)
    
    # Compute partial sum for this block
    # sum += a[i] + b[i] for each i
    partial_sum = tl.sum(tl.where(mask, a_vals + b_vals, 0.0))
    
    # Store partial sum
    tl.store(sum_ptr + pid, partial_sum)

def s319_triton(a, b, c, d, e):
    N = a.shape[0]
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    # Create tensor to store partial sums
    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s319_kernel[grid](
        a, b, c, d, e,
        N,
        partial_sums,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reduce partial sums to get final sum
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()