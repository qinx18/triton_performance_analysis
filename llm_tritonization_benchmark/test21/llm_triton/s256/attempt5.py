import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, j, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid = tl.program_id(0)
    
    # Calculate block start and offsets for i dimension
    block_start = pid * BLOCK_SIZE
    i_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N
    
    # Load a[j] (which was already updated)
    a_j_val = tl.load(a_ptr + j)
    
    # Load bb[j][i] for all i in block
    bb_offsets = j * N + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
    
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
    
    # Loop over i from 0 to N-1
    for i in range(N):
        # Loop over j sequentially from 1 to N-1
        for j in range(1, N):
            # Update a[j] first
            a[j] = 1.0 - a[j-1]
            
            # Update aa[j][i] = a[j] + bb[j][i]*d[j]
            aa[j, i] = a[j] + bb[j, i] * d[j]