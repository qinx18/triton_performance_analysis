import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, output_ptr, len_2d, inc, j, aa_stride_0, aa_stride_1, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < (len_2d - 1)
    
    # Load ip values
    ip_vals = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
    
    # Compute off = inc + i
    off_vals = inc + current_offsets
    
    # Load a[off] values
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Compute aa indices: aa[j-1][ip[i]]
    aa_row_offset = (j - 1) * aa_stride_0
    aa_col_offsets = ip_vals * aa_stride_1
    aa_offsets = aa_row_offset + aa_col_offsets
    
    # Load aa values
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Store products for reduction
    tl.store(output_ptr + current_offsets, products, mask=mask)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    n = len_2d - 1
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor for partial products with proper size
    output = torch.zeros(triton.cdiv(n, BLOCK_SIZE) * BLOCK_SIZE, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s4116_kernel[(num_blocks,)](
        a, aa, ip, output, 
        len_2d, inc, j,
        aa.stride(0), aa.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum only the valid elements
    return output[:n].sum().item()