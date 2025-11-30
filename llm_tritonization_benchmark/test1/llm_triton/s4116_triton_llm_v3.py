import torch
import triton
import triton.language as tl

@triton.jit
def s4116_kernel(a_ptr, aa_ptr, ip_ptr, inc, j, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    if pid > 0:
        return
    
    sum_val = 0.0
    
    # Process in blocks
    for block_start in range(0, len_2d - 1, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, len_2d - 1)
        block_size = block_end - block_start
        
        # Create offsets for this block
        i_offsets = tl.arange(0, BLOCK_SIZE) + block_start
        mask = tl.arange(0, BLOCK_SIZE) < block_size
        
        # Load indices with masking
        ip_indices = tl.load(ip_ptr + i_offsets, mask=mask, other=0)
        
        # Calculate offsets for array a
        a_offsets = inc + i_offsets
        
        # Load from array a
        a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)
        
        # Calculate aa indices: aa[j-1][ip[i]]
        aa_indices = (j - 1) * len_2d + ip_indices
        
        # Load from aa array
        aa_vals = tl.load(aa_ptr + aa_indices, mask=mask, other=0.0)
        
        # Compute products and accumulate
        products = a_vals * aa_vals
        sum_val += tl.sum(products, axis=0)
    
    # Store result (only one thread writes)
    tl.store(tl.program_id(0) * 1 + tl.arange(0, 1), sum_val)

def s4116_triton(a, aa, ip, inc, j):
    len_2d = aa.shape[0]
    
    # Create output tensor for the sum
    output = torch.zeros(1, dtype=torch.float32, device=a.device)
    
    BLOCK_SIZE = 256
    grid = (1,)
    
    s4116_kernel[grid](
        a, aa, ip, inc, j, len_2d, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output.item()