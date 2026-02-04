import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process anti-diagonals sequentially inside kernel
    for diag in range(2, 2 * N):
        j = diag - i_offsets
        
        # Mask for valid positions
        mask = (j >= 1) & (j < N) & (i_offsets >= 1) & (i_offsets < N)
        
        # Calculate linear indices
        aa_idx = j * N + i_offsets
        aa_left_idx = j * N + (i_offsets - 1)
        aa_up_idx = (j - 1) * N + i_offsets
        
        # Load values
        aa_left = tl.load(aa_ptr + aa_left_idx, mask=mask, other=0.0)
        aa_up = tl.load(aa_ptr + aa_up_idx, mask=mask, other=0.0)
        
        # Compute new value
        result = (aa_left + aa_up) / 1.9
        
        # Store result
        tl.store(aa_ptr + aa_idx, result, mask=mask)
        
        # Memory barrier to ensure writes are visible before next diagonal
        tl.debug_barrier()

def s2111_triton(aa, len_2d):
    N = len_2d
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    s2111_kernel[grid](aa, N, BLOCK_SIZE)