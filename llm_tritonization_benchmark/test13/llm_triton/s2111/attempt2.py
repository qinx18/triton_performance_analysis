import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, diag, start_j, end_j, N, BLOCK_SIZE: tl.constexpr):
    thread_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Check if this thread should process an element
    mask = thread_idx < (end_j - start_j)
    
    # Calculate j and i coordinates
    j = start_j + thread_idx
    i = diag - j
    
    # Additional bounds check
    valid = mask & (j >= 1) & (j < N) & (i >= 1) & (i < N)
    
    # Load values
    left_offset = j * N + (i - 1)
    top_offset = (j - 1) * N + i
    current_offset = j * N + i
    
    left_val = tl.load(aa_ptr + left_offset, mask=valid)
    top_val = tl.load(aa_ptr + top_offset, mask=valid)
    
    # Compute new value
    new_val = (left_val + top_val) / 1.9
    
    # Store result
    tl.store(aa_ptr + current_offset, new_val, mask=valid)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process each anti-diagonal sequentially
    for diag in range(2, 2 * N - 1):
        start_j = max(1, diag - N + 1)
        end_j = min(diag + 1, N)
        num_elements = end_j - start_j
        
        if num_elements > 0:
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            s2111_kernel[grid](aa, diag, start_j, end_j, N, BLOCK_SIZE)