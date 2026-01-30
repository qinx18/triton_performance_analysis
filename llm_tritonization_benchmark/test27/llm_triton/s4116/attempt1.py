import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, sum_ptr, 
                 len_2d, inc, j,
                 BLOCK_SIZE: tl.constexpr):
    
    # Calculate offsets for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid indices
    mask = i_offsets < (len_2d - 1)
    
    # Calculate off = inc + i (scalar expansion)
    off_values = inc + i_offsets
    
    # Load ip[i] values
    ip_indices = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    
    # Load a[off] values
    a_values = tl.load(a_ptr + off_values, mask=mask, other=0.0)
    
    # Calculate aa indices: (j-1) * len_2d + ip[i]
    aa_indices = (j - 1) * len_2d + ip_indices
    
    # Load aa[j-1][ip[i]] values
    aa_values = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = tl.where(mask, a_values * aa_values, 0.0)
    
    # Sum the products
    partial_sum = tl.sum(products)
    
    # Store partial sum
    pid = tl.program_id(0)
    tl.store(sum_ptr + pid, partial_sum)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    BLOCK_SIZE = 256
    n = len_2d - 1
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create tensor to store partial sums
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s4116_kernel[grid](
        a, aa, ip, partial_sums,
        len_2d, inc, j,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results
    total_sum = torch.sum(partial_sums)
    return total_sum.item()