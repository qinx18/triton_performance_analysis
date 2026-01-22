import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, diag: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the element index for this anti-diagonal
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate range of j indices for this diagonal
    start_j = tl.maximum(1, diag - N + 1)
    end_j = tl.minimum(diag, N)
    
    # Calculate j and i from the flat index
    j = start_j + idx
    i = diag - j
    
    # Mask for valid indices
    mask = (idx < (end_j - start_j)) & (j >= 1) & (j < N) & (i >= 1) & (i < N)
    
    # Calculate linear indices
    current_idx = j * N + i
    left_idx = j * N + (i - 1)
    up_idx = (j - 1) * N + i
    
    # Load values
    left_val = tl.load(aa_ptr + left_idx, mask=mask, other=0.0)
    up_val = tl.load(aa_ptr + up_idx, mask=mask, other=0.0)
    
    # Compute new value
    new_val = (left_val + up_val) / 1.9
    
    # Store result
    tl.store(aa_ptr + current_idx, new_val, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        
        if start_j >= end_j:
            continue
            
        num_elements = end_j - start_j
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s2111_kernel[grid](
            aa, 
            N=N, 
            diag=diag, 
            BLOCK_SIZE=BLOCK_SIZE
        )