import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    for j in range(1, N):
        # Current row addresses
        curr_aa_ptrs = aa_ptr + j * N + i_idx
        prev_aa_ptrs = aa_ptr + (j - 1) * N + i_idx  
        bb_ptrs = bb_ptr + j * N + i_idx
        
        # Load previous aa values and bb values with proper masking
        prev_aa_vals = tl.load(prev_aa_ptrs, mask=i_mask, other=0.0)
        bb_vals = tl.load(bb_ptrs, mask=i_mask, other=0.0)
        
        # Compute new aa values
        new_aa_vals = prev_aa_vals + bb_vals
        
        # Store results with proper masking
        tl.store(curr_aa_ptrs, new_aa_vals, mask=i_mask)

def s231_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s231_kernel[grid](aa, bb, N, BLOCK_SIZE)
    return aa