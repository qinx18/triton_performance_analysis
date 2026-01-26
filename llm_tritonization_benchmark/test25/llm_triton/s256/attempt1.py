import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension (parallelized)
    i = tl.program_id(0)
    
    if i >= len_2d:
        return
    
    # Sequential loop over j dimension
    for j in range(1, len_2d):
        # Load a[j-1] (scalar)
        a_prev = tl.load(a_ptr + (j - 1))
        
        # Compute a[j] = 1.0 - a[j-1]
        a_val = 1.0 - a_prev
        
        # Store a[j]
        tl.store(a_ptr + j, a_val)
        
        # Load bb[j][i] and d[j] for aa computation
        bb_val = tl.load(bb_ptr + j * len_2d + i)
        d_val = tl.load(d_ptr + j)
        
        # Compute aa[j][i] = a[j] + bb[j][i]*d[j]
        aa_val = a_val + bb_val * d_val
        
        # Store aa[j][i]
        tl.store(aa_ptr + j * len_2d + i, aa_val)

def s256_triton(a, aa, bb, d, len_2d):
    # Get dimensions
    N = len_2d
    
    # Define block size
    BLOCK_SIZE = 128
    
    # Calculate grid size for i dimension
    grid_size = triton.cdiv(N, 1)  # One thread per i
    
    # Launch kernel
    s256_kernel[(grid_size,)](
        a, aa, bb, d, N, BLOCK_SIZE
    )