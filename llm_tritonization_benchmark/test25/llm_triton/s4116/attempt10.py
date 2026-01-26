import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, output_ptr, n, inc, j, aa_stride_0, aa_stride_1, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_vals = block_start + offsets
    mask = i_vals < n
    
    # Load ip[i] values
    ip_vals = tl.load(ip_ptr + i_vals, mask=mask, other=0)
    
    # Compute off = inc + i
    off_vals = inc + i_vals
    
    # Load a[off] values
    a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
    
    # Compute aa[j-1][ip[i]] indices
    aa_row_offset = (j - 1) * aa_stride_0
    aa_col_offsets = ip_vals * aa_stride_1
    aa_offsets = aa_row_offset + aa_col_offsets
    
    # Load aa values
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    
    # Compute products
    products = a_vals * aa_vals
    
    # Store products
    tl.store(output_ptr + i_vals, products, mask=mask)

def s4116_triton(a, aa, ip, len_2d, inc, j):
    n = len_2d - 1
    BLOCK_SIZE = 256
    num_blocks = triton.cdiv(n, BLOCK_SIZE)
    
    # Create output tensor
    output = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    # Launch kernel
    s4116_kernel[(num_blocks,)](
        a, aa, ip, output, 
        n, inc, j,
        aa.stride(0), aa.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Sum all products
    return output.sum().item()