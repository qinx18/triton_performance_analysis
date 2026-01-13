import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, n_elements, aa_cols, output_ptr, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Mask for valid indices
    mask = indices < n_elements
    
    # Load ip values for this block
    ip_vals = tl.load(ip_ptr + indices, mask=mask, other=0)
    
    # Compute off values: off = inc + i
    off_vals = inc + indices
    
    # Load a[off] values
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Compute aa indices: (j-1) * aa_cols + ip[i]
    aa_row_offset = (j - 1) * aa_cols
    aa_indices = aa_row_offset + ip_vals
    
    # Load aa values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Apply mask to products to zero out invalid elements
    products = tl.where(mask, products, 0.0)
    
    # Store products for reduction
    tl.store(output_ptr + indices, products, mask=mask)

def s4116_triton(a, aa, ip, inc, j):
    n_elements = aa.shape[0] - 1  # LEN_2D - 1
    aa_cols = aa.shape[1]
    
    # Create output tensor for partial products
    partial_products = torch.zeros(n_elements, device=a.device, dtype=a.dtype)
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s4116_kernel[grid](
        a, aa, ip, inc, j, n_elements, aa_cols, partial_products, BLOCK_SIZE
    )
    
    # Sum all partial products
    return partial_products.sum().item()