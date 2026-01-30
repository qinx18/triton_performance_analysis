import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, len_2d, inc, j, aa_stride0, aa_stride1, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < (len_2d - 1)
    
    # Load ip values with bounds checking
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute off = inc + i
    off_vals = inc + indices
    
    # Load a[off] values with bounds checking
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Compute aa indices: (j-1) * aa_stride0 + ip[i] * aa_stride1
    aa_row_offset = (j - 1) * aa_stride0
    aa_col_offset = ip_vals * aa_stride1
    aa_indices = aa_row_offset + aa_col_offset
    
    # Create additional mask for valid ip values
    ip_mask = (ip_vals >= 0) & (ip_vals < len_2d)
    final_mask = mask & ip_mask
    
    # Load aa values with bounds checking
    aa_vals = tl.load(aa_ptr + aa_indices, mask=final_mask, other=0.0)
    
    # Compute products
    products = tl.where(final_mask, a_vals * aa_vals, 0.0)
    
    # Return sum of products for this block
    return tl.sum(products)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    n_elements = len_2d - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Get strides
    aa_stride0 = aa.stride(0)
    aa_stride1 = aa.stride(1)
    
    # Create output tensor for partial sums
    partial_sums = torch.zeros(grid[0], dtype=torch.float32, device=a.device)
    
    # Launch kernels
    s4116_kernel[grid](
        a, aa, ip, 
        len_2d, inc, j,
        aa_stride0, aa_stride1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Manual reduction since kernel returns aren't supported
    total_sum = 0.0
    for block_id in range(grid[0]):
        block_start = block_id * BLOCK_SIZE
        block_end = min(block_start + BLOCK_SIZE, n_elements)
        
        if block_start < n_elements:
            indices = torch.arange(block_start, block_end, device=a.device)
            
            # Get ip values and check bounds
            ip_vals = ip[indices]
            valid_ip = (ip_vals >= 0) & (ip_vals < len_2d)
            
            # Compute off values
            off_vals = inc + indices
            
            # Get values with bounds checking
            a_vals = a[off_vals]
            aa_vals = aa[j-1, ip_vals.clamp(0, len_2d-1)]
            
            # Apply mask and compute sum
            products = torch.where(valid_ip, a_vals * aa_vals, 0.0)
            total_sum += products.sum().item()
    
    return total_sum