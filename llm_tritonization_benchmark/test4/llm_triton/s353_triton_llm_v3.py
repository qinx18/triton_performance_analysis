import triton
import triton.language as tl
import torch

@triton.jit
def s353_kernel(a_ptr, b_ptr, ip_ptr, alpha, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(0, 5):
        current_offsets = block_start + offsets + i
        mask = current_offsets < n
        
        # Load indices and gather from b
        ip_indices = tl.load(ip_ptr + current_offsets, mask=mask)
        b_values = tl.load(b_ptr + ip_indices, mask=mask)
        
        # Load a values
        a_values = tl.load(a_ptr + current_offsets, mask=mask)
        
        # Compute saxpy and store back
        result = a_values + alpha * b_values
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s353_triton(a, b, c, ip):
    alpha = c[0].item()
    n = a.shape[0]
    
    # Process elements in groups of 5, so we need n//5 groups
    num_groups = (n + 4) // 5  # Round up division
    
    BLOCK_SIZE = 128
    grid = (triton.cdiv(num_groups, BLOCK_SIZE),)
    
    # Adjust n to be the number of groups * 5
    effective_n = num_groups * 5
    
    s353_kernel[grid](
        a, b, ip, alpha, effective_n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a