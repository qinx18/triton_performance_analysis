import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    thread_id = tl.arange(0, BLOCK_SIZE)
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N):
        # For each thread, calculate i and j
        i = thread_id
        j = diag - i
        
        # Create mask for valid (i, j) pairs
        mask = (i >= 1) & (i < N) & (j >= 1) & (j < N)
        
        # Calculate offsets for current, left, and top positions
        current_offset = j * N + i
        left_offset = j * N + (i - 1)  # aa[j][i-1]
        top_offset = (j - 1) * N + i   # aa[j-1][i]
        
        # Load values
        left_val = tl.load(aa_ptr + left_offset, mask=mask, other=0.0)
        top_val = tl.load(aa_ptr + top_offset, mask=mask, other=0.0)
        
        # Compute result
        result = (left_val + top_val) / 1.9
        
        # Store result
        tl.store(aa_ptr + current_offset, result, mask=mask)

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = 512
    
    grid = (1,)
    s2111_kernel[grid](aa, N=N, BLOCK_SIZE=BLOCK_SIZE)