import triton
import triton.language as tl
import torch

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, len_2d, 
                 output_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    # Initialize sum
    sum_val = 0.0
    
    # Process in blocks
    for block_start in range(0, len_2d - 1, BLOCK_SIZE):
        offsets = tl.arange(0, BLOCK_SIZE)
        i_vals = block_start + offsets
        mask = (block_start + offsets) < (len_2d - 1)
        
        # Compute off = inc + i
        off_vals = inc + i_vals
        
        # Load a[off] values
        a_vals = tl.load(a_ptr + off_vals, mask=mask, other=0.0)
        
        # Load ip[i] values
        ip_vals = tl.load(ip_ptr + i_vals, mask=mask, other=0)
        
        # Compute aa indices: (j-1) * len_2d + ip[i]
        aa_indices = (j - 1) * len_2d + ip_vals
        
        # Load aa[j-1][ip[i]] values
        aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
        
        # Compute products and sum
        products = a_vals * aa_vals
        
        # Sum the valid products
        masked_products = tl.where(mask, products, 0.0)
        block_sum = tl.sum(masked_products)
        sum_val += block_sum
    
    # Store result
    tl.store(output_ptr, sum_val)

def s4116_triton(a, aa, ip, inc, j):
    len_2d = aa.shape[0]
    
    # Create output tensor
    output = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    # Launch kernel with single program
    BLOCK_SIZE = 256
    grid = (1,)
    
    s4116_kernel[grid](
        a, aa, ip,
        inc, j, len_2d,
        output,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()