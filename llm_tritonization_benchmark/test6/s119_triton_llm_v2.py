import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, N, i_offset, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid_j = tl.program_id(0)
    
    # Calculate j indices for this block
    j_start = pid_j * BLOCK_SIZE + 1  # Start from j=1
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid j indices
    j_mask = j_offsets < N
    
    # Current i index
    i = i_offset + 1  # Start from i=1
    
    # Calculate memory offsets
    # Read from aa[i-1, j-1]
    read_offsets = (i - 1) * N + (j_offsets - 1)
    # Write to aa[i, j]
    write_offsets = i * N + j_offsets
    # Read from bb[i, j]
    bb_offsets = i * N + j_offsets
    
    # Load data
    aa_prev = tl.load(aa_ptr + read_offsets, mask=j_mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    
    # Compute result
    result = aa_prev + bb_vals
    
    # Store result
    tl.store(aa_ptr + write_offsets, result, mask=j_mask)

def s119_triton(aa, bb):
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i dimension (i from 1 to N-1)
    for i in range(N - 1):
        # Number of j elements to process (j from 1 to N-1)
        num_j = N - 1
        if num_j > 0:
            grid = (triton.cdiv(num_j, BLOCK_SIZE),)
            s119_kernel[grid](aa, bb, N, i, BLOCK_SIZE=BLOCK_SIZE)
    
    return aa