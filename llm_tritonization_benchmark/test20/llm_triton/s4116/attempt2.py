import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, len_2d_minus_1, output_ptr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block offsets
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    
    # Mask for valid indices
    mask = i_offsets < len_2d_minus_1
    
    # Load ip[i] values
    ip_vals = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
    
    # Calculate a[off] indices (off = inc + i)
    a_indices = inc + i_offsets
    
    # Load a[off] values
    a_vals = tl.load(a_ptr + a_indices, mask=mask, other=0.0)
    
    # Calculate aa[j-1][ip[i]] indices (2D array flattened)
    aa_row_start = (j - 1) * len_2d_minus_1  # Note: using len_2d_minus_1 as row size
    aa_indices = aa_row_start + ip_vals
    
    # Load aa[j-1][ip[i]] values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Calculate products and sum
    products = a_vals * aa_vals
    
    # Zero out invalid products
    products = tl.where(mask, products, 0.0)
    
    # Sum the products
    partial_sum = tl.sum(products)
    
    # Store the partial sum
    tl.store(output_ptr + pid, partial_sum)

def s4116_triton(a, aa, ip, inc, j, len_2d):
    # Calculate actual loop bound
    len_2d_minus_1 = len_2d - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d_minus_1, BLOCK_SIZE),)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid[0], dtype=torch.float32, device=a.device)
    
    # Launch kernel
    s4116_kernel[grid](
        a, aa, ip, inc, j, len_2d_minus_1, partial_sums,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()