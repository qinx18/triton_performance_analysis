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
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute off = inc + i
    off_vals = inc + indices
    
    # Load a[off] values  
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Compute aa indices: (j-1) * aa_stride0 + ip[i] * aa_stride1
    aa_row_offset = (j - 1) * aa_stride0
    aa_col_offset = ip_vals * aa_stride1
    aa_indices = aa_row_offset + aa_col_offset
    
    # Load aa values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = tl.where(mask, a_vals * aa_vals, 0.0)
    
    # Return sum of products for this block
    return tl.sum(products)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    n_elements = len_2d - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Get strides
    aa_stride0 = aa.stride(0)
    aa_stride1 = aa.stride(1)
    
    # Create output tensor for partial sums from each block
    block_sums = torch.zeros(grid[0], dtype=a.dtype, device=a.device)
    
    # Launch kernel to compute block sums
    for block_id in range(grid[0]):
        block_start = block_id * BLOCK_SIZE
        block_offsets = torch.arange(BLOCK_SIZE, device=a.device)
        indices = block_start + block_offsets
        mask = indices < n_elements
        
        if mask.any():
            # Load ip values
            ip_vals = torch.where(mask, ip[indices], 0)
            
            # Compute off = inc + i
            off_vals = inc + indices
            
            # Load a[off] values
            a_vals = torch.where(mask, a[off_vals], 0.0)
            
            # Load aa values
            aa_vals = torch.where(mask, aa[j-1, ip_vals], 0.0)
            
            # Compute products and sum
            products = torch.where(mask, a_vals * aa_vals, 0.0)
            block_sums[block_id] = products.sum()
    
    # Sum all block results
    return block_sums.sum().item()