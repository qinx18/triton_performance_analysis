import torch
import triton
import triton.language as tl

@triton.jit
def s353_kernel(a_ptr, b_ptr, alpha, ip_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and calculate base offset
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process 5 elements at a time within each block
    for i in range(0, BLOCK_SIZE, 5):
        current_offsets = block_start + i + offsets[:5]
        mask = current_offsets < n_elements
        
        # Load indices from ip array
        ip_indices = tl.load(ip_ptr + current_offsets, mask=mask, other=0)
        
        # Load values from b array using gathered indices
        b_vals = tl.load(b_ptr + ip_indices, mask=mask, other=0.0)
        
        # Load current values from a array
        a_vals = tl.load(a_ptr + current_offsets, mask=mask, other=0.0)
        
        # Compute saxpy: a[i] += alpha * b[ip[i]]
        result = a_vals + alpha * b_vals
        
        # Store back to a array
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s353_triton(a, b, c, ip):
    n_elements = a.shape[0]
    alpha = c[0]
    
    # Use block size that's divisible by 5 for the unrolled loop
    BLOCK_SIZE = 320  # 64 * 5
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s353_kernel[grid](
        a, b, alpha, ip, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )