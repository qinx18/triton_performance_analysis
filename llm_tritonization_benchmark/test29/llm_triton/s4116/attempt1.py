import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, 
                 len_2d, inc, j,
                 output_ptr,
                 aa_stride_0, aa_stride_1,
                 BLOCK_SIZE: tl.constexpr):
    
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < (len_2d - 1)
    
    # Load indices and compute array access patterns
    ip_vals = tl.load(ip_ptr + idx, mask=mask, other=0)
    
    # Compute off = inc + i
    off_vals = inc + idx
    
    # Load a[off] values
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Compute aa access: aa[j-1][ip[i]]
    aa_row = j - 1
    aa_offsets = aa_row * aa_stride_0 + ip_vals * aa_stride_1
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products
    products = tl.where(mask, a_vals * aa_vals, 0.0)
    
    # Store products for reduction
    tl.store(output_ptr + idx, products, mask=mask)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    n = len_2d - 1
    BLOCK_SIZE = 256
    
    # Create output tensor for partial products
    products = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Calculate grid size
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    s4116_kernel[grid](
        a, aa, ip,
        len_2d, inc, j,
        products,
        aa.stride(0), aa.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products to get final result
    return products.sum().item()