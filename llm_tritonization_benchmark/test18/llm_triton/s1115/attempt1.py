import torch
import triton
import triton.language as tl

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, n_size, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid_j = tl.program_id(0)
    
    # Calculate j offsets for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid_j * BLOCK_SIZE + j_offsets
    j_mask = j_idx < n_size
    
    # Sequential loop over i dimension inside kernel
    for i in range(n_size):
        # Calculate base addresses for this i iteration
        aa_base = aa_ptr + i * n_size
        bb_base = bb_ptr + i * n_size
        cc_base = cc_ptr + j_idx * n_size + i  # Note: cc[j][i] indexing
        
        # Load data
        aa_vals = tl.load(aa_base + j_idx, mask=j_mask)
        bb_vals = tl.load(bb_base + j_idx, mask=j_mask)
        cc_vals = tl.load(cc_base, mask=j_mask)
        
        # Compute: aa[i][j] = aa[i][j] * cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store result
        tl.store(aa_base + j_idx, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    n_size = aa.shape[0]
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_size, BLOCK_SIZE),)
    
    s1115_kernel[grid](
        aa, bb, cc,
        n_size,
        BLOCK_SIZE=BLOCK_SIZE
    )