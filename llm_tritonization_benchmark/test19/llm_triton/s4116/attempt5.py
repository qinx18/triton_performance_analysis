import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, len_2d, output_ptr,
                 BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    
    # Mask for valid indices (loop goes to len_2d-1, so i < len_2d-1)
    mask = i_indices < (len_2d - 1)
    
    # Load ip values for this block
    ip_vals = tl.load(ip_ptr + i_indices, mask=mask, other=0)
    
    # Compute a indices: off = inc + i
    a_indices = inc + i_indices
    
    # Load a values
    a_vals = tl.load(a_ptr + a_indices, mask=mask, other=0.0)
    
    # Compute aa indices: aa[j-1][ip[i]]
    # aa is stored as flat array, so aa[row][col] = aa[row * len_2d + col]
    aa_row = j - 1
    aa_col_indices = ip_vals
    aa_indices = aa_row * len_2d + aa_col_indices
    
    # Load aa values - need to mask based on valid ip values and bounds
    ip_mask = (ip_vals >= 0) & (ip_vals < len_2d)
    combined_mask = mask & ip_mask
    aa_vals = tl.load(aa_ptr + aa_indices, mask=combined_mask, other=0.0)
    
    # Compute products - only for valid indices
    products = tl.where(combined_mask, a_vals * aa_vals, 0.0)
    
    # Sum the products in this block
    block_sum = tl.sum(products)
    
    # Store partial sum (each thread block stores one value)
    tl.store(output_ptr + pid, block_sum)

def s4116_triton(a, aa, ip, inc, j, len_2d):
    # Ensure tensors are on GPU
    device = a.device
    
    BLOCK_SIZE = 256
    num_elements = len_2d - 1
    num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=device)
    
    # Launch kernel
    grid = (num_blocks,)
    s4116_kernel[grid](
        a, aa, ip, inc, j, len_2d, partial_sums,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all partial results
    total_sum = torch.sum(partial_sums)
    
    return total_sum.item()