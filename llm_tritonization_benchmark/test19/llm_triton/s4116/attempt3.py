import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, n, aa_stride_0, aa_stride_1, 
                 partial_sums_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n
    
    # Load indices
    ip_vals = tl.load(ip_ptr + idx, mask=mask, other=0)
    
    # Compute array offsets
    a_offsets = inc + idx
    aa_offsets = (j - 1) * aa_stride_0 + ip_vals * aa_stride_1
    
    # Load values
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products and sum
    products = a_vals * aa_vals
    block_sum = tl.sum(tl.where(mask, products, 0.0))
    
    # Store partial sum
    tl.store(partial_sums_ptr + pid, block_sum)

def s4116_triton(a, aa, ip, inc, j):
    n = aa.shape[0] - 1
    aa_stride_0 = aa.stride(0)
    aa_stride_1 = aa.stride(1)
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid[0], dtype=torch.float32, device=a.device)
    
    # Launch kernel
    s4116_kernel[grid](
        a, aa, ip, inc, j, n, aa_stride_0, aa_stride_1, 
        partial_sums, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results and return as scalar
    total_sum = partial_sums.sum()
    return total_sum