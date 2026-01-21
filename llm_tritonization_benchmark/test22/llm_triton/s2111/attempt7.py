import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr):
    # Get diagonal index
    diag_idx = tl.program_id(0)
    diag = diag_idx + 2  # Start from diagonal 2 (j=1, i=1)
    
    # Calculate bounds for this diagonal
    start_j = tl.maximum(1, diag - N + 1)
    end_j = tl.minimum(diag, N)
    
    # Skip if no valid elements on this diagonal
    if start_j >= end_j:
        return
    
    # Process all elements on this diagonal
    for j in range(start_j, end_j):
        i = diag - j
        valid_i = (i >= 1) & (i < N)
        valid_j = (j >= 1) & (j < N)
        if valid_i & valid_j:
            # Load previous values
            val_left = tl.load(aa_ptr + j * N + (i - 1))
            val_up = tl.load(aa_ptr + (j - 1) * N + i)
            
            # Compute and store result
            result = (val_left + val_up) / 1.9
            tl.store(aa_ptr + j * N + i, result)

def s2111_triton(aa):
    N = aa.shape[0]
    
    # Total number of diagonals to process
    # Diagonals range from 2 to 2*N-2 (for indices starting from 1)
    num_diagonals = 2 * N - 3
    
    if num_diagonals > 0:
        grid = (num_diagonals,)
        s2111_kernel[grid](aa, N=N)
    
    return aa