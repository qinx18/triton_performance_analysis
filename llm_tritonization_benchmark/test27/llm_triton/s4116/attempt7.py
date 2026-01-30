import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, sum_ptr, len_2d, inc, j, aa_stride0, BLOCK_SIZE: tl.constexpr):
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
    
    # Compute aa indices: (j-1) * aa_stride0 + ip[i]
    aa_row_offset = (j - 1) * aa_stride0
    aa_indices = aa_row_offset + ip_vals
    
    # Load aa values - need mask for valid ip values
    ip_mask = mask & (ip_vals >= 0) & (ip_vals < aa_stride0)
    aa_vals = tl.load(aa_ptr + aa_indices, mask=ip_mask, other=0.0)
    
    # Compute products - use original mask to ensure we only compute for valid i
    products = tl.where(mask, a_vals * aa_vals, 0.0)
    
    # Store products for reduction
    tl.store(sum_ptr + indices, products, mask=mask)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    n_elements = len_2d - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create output tensor for partial sums
    sum_tensor = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Get strides
    aa_stride0 = aa.stride(0)
    
    # Launch kernel
    s4116_kernel[grid](
        a, aa, ip, sum_tensor,
        len_2d, inc, j, aa_stride0,
        BLOCK_SIZE
    )
    
    # Sum all elements
    return sum_tensor.sum().item()