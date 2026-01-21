import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, len_2d, 
                 output_ptr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block offsets
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices
    mask = indices < (len_2d - 1)
    
    # Initialize sum for this block
    block_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load ip indices
    ip_indices = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate a array offsets: off = inc + i
    a_offsets = inc + indices
    
    # Load from a array
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Calculate aa array offsets: aa[j-1][ip[i]]
    aa_row_offset = (j - 1) * len_2d
    aa_offsets = aa_row_offset + ip_indices
    
    # Load from aa array
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products
    products = tl.where(mask, a_vals * aa_vals, 0.0)
    
    # Sum the products
    block_sum = tl.sum(products)
    
    # Store the block sum
    tl.store(output_ptr + pid, block_sum)

def s4116_triton(a, aa, ip, inc, j, len_2d):
    # Determine block size and grid
    BLOCK_SIZE = 256
    n_elements = len_2d - 1
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid_size, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    grid = (grid_size,)
    s4116_kernel[grid](
        a, aa, ip, inc, j, len_2d,
        partial_sums, 
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()