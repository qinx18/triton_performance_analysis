import torch
import triton
import triton.language as tl

@triton.jit
def s351_kernel(a_ptr, b_ptr, alpha, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process 5 elements at a time, starting from pid * BLOCK_SIZE
    base_idx = pid * BLOCK_SIZE
    
    for block_start in range(base_idx, base_idx + BLOCK_SIZE, 5):
        # Load 5 consecutive elements
        idx_offsets = block_start + tl.arange(0, 5)
        mask = idx_offsets < n_elements
        
        # Load a and b values
        a_vals = tl.load(a_ptr + idx_offsets, mask=mask)
        b_vals = tl.load(b_ptr + idx_offsets, mask=mask)
        
        # Compute saxpy: a[i] += alpha * b[i]
        result = a_vals + alpha * b_vals
        
        # Store back to a
        tl.store(a_ptr + idx_offsets, result, mask=mask)

def s351_triton(a, b, c):
    n_elements = a.numel()
    alpha = c[0].item()
    
    # Use block size that's a multiple of 5 for efficiency
    BLOCK_SIZE = 320  # 64 * 5, ensures each block processes complete groups of 5
    
    # Calculate grid size to cover all elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s351_kernel[grid](
        a, b, alpha, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )