import torch
import triton
import triton.language as tl

@triton.jit
def s111_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for start_idx in range(1, n_elements, 2 * BLOCK_SIZE):
        current_offsets = start_idx + block_start + offsets * 2
        mask = current_offsets < n_elements
        
        # Load a[i-1] and b[i]
        a_prev_offsets = current_offsets - 1
        a_prev_mask = (a_prev_offsets >= 0) & mask
        b_mask = mask
        
        a_prev = tl.load(a_ptr + a_prev_offsets, mask=a_prev_mask, other=0.0)
        b_vals = tl.load(b_ptr + current_offsets, mask=b_mask, other=0.0)
        
        # Compute a[i] = a[i-1] + b[i]
        result = a_prev + b_vals
        
        # Store result
        tl.store(a_ptr + current_offsets, result, mask=mask)

def s111_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate number of odd indices: (1, 3, 5, ..., n-1 or n-2)
    num_odd_indices = (n_elements - 1 + 1) // 2
    grid = (triton.cdiv(num_odd_indices, BLOCK_SIZE),)
    
    s111_kernel[grid](a, b, n_elements, BLOCK_SIZE)