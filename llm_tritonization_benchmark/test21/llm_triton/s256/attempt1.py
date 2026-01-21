import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid = tl.program_id(0)
    
    # Calculate block start and offsets for i dimension
    block_start = pid * BLOCK_SIZE
    i_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N
    
    # For each j from 1 to N
    for j in range(1, N):
        # Load a[j-1] (scalar load)
        a_j_minus_1 = tl.load(a_ptr + (j - 1))
        
        # Compute a[j] = 1.0 - a[j-1]
        a_j_val = 1.0 - a_j_minus_1
        
        # Store a[j] (scalar store)
        tl.store(a_ptr + j, a_j_val)
        
        # Load bb[j][i] for all i in block
        bb_offsets = j * N + i_offsets
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
        
        # Load d[j] (scalar load)
        d_j_val = tl.load(d_ptr + j)
        
        # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
        aa_vals = a_j_val + bb_vals * d_j_val
        
        # Store aa[j][i] for all i in block
        aa_offsets = j * N + i_offsets
        tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # Launch kernel
    s256_kernel[grid](
        a, aa, bb, d, N,
        BLOCK_SIZE=BLOCK_SIZE
    )