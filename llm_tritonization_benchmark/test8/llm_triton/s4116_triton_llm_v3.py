import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, LEN_2D, output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    sum_val = 0.0
    
    for block_idx in range(0, LEN_2D - 1, BLOCK_SIZE):
        current_offsets = block_idx + offsets
        mask = (current_offsets < LEN_2D - 1)
        
        # Load indices
        ip_indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Calculate off = inc + i
        off_indices = inc + current_offsets
        
        # Load a[off]
        a_vals = tl.load(a_ptr + off_indices, mask=mask, other=0.0)
        
        # Calculate aa indices: (j-1) * LEN_2D + ip[i]
        aa_indices = (j - 1) * LEN_2D + ip_indices
        aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
        
        # sum += a[off] * aa[j-1][ip[i]]
        products = a_vals * aa_vals
        masked_products = tl.where(mask, products, 0.0)
        sum_val += tl.sum(masked_products)
    
    if pid == 0:
        tl.store(output_ptr, sum_val)

def s4116_triton(a, aa, ip, inc, j):
    LEN_2D = aa.shape[0]
    
    # Create output tensor
    output = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s4116_kernel[grid](
        a, aa, ip, inc, j, LEN_2D, output, BLOCK_SIZE
    )
    
    return output.item()