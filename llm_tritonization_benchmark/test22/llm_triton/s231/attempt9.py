import triton
import triton.language as tl

@triton.jit
def s231_kernel(aa, bb, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    mask_i = i_idx < N
    
    for j in range(1, N):
        # Load aa[j-1][i] for all i in this block
        prev_j_ptrs = aa + (j - 1) * N + i_idx
        prev_vals = tl.load(prev_j_ptrs, mask=mask_i, other=0.0)
        
        # Load bb[j][i] for all i in this block  
        bb_ptrs = bb + j * N + i_idx
        bb_vals = tl.load(bb_ptrs, mask=mask_i, other=0.0)
        
        # Compute aa[j][i] = aa[j-1][i] + bb[j][i]
        result = prev_vals + bb_vals
        
        # Store to aa[j][i]
        curr_j_ptrs = aa + j * N + i_idx
        tl.store(curr_j_ptrs, result, mask=mask_i)
        
        # Ensure memory ordering - wait for stores to complete before next iteration
        tl.debug_barrier()

def s231_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s231_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)