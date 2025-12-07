import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, n_elements, output_ptr,
                 BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < n_elements
    
    # Load indirect indices with bounds checking
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Calculate offsets for array a
    a_offsets = inc + indices
    
    # Load values from array a
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
    
    # Calculate 2D indices for aa array (j-1 is the row)
    aa_row = j - 1
    aa_indices = aa_row * 256 + ip_vals  # aa[j-1][ip[i]]
    
    # Load values from aa array
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Apply mask to products
    masked_products = tl.where(mask, products, 0.0)
    
    # Sum the products
    block_sum = tl.sum(masked_products)
    
    # Store the result
    tl.store(output_ptr + pid, block_sum)

def s4116_triton(a, aa, ip, inc, j):
    # Get dimensions
    n_elements = 255  # LEN_2D - 1
    
    # Choose block size
    BLOCK_SIZE = 64
    
    # Calculate number of blocks
    num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(num_blocks, dtype=torch.float32, device=a.device)
    
    # Launch kernel
    grid = (num_blocks,)
    s4116_kernel[grid](
        a, aa, ip, inc, j, n_elements, partial_sums,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum the partial results and return as Python scalar
    total_sum = torch.sum(partial_sums)
    
    return float(total_sum.item())