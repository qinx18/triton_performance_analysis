import triton
import triton.language as tl
import torch

@triton.jit
def s1115_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    for i in range(N):
        # Calculate pointers for row i
        aa_row_ptr = aa_ptr + i * N
        bb_row_ptr = bb_ptr + i * N
        
        # Load data
        aa_vals = tl.load(aa_row_ptr + j_idx, mask=j_mask)
        bb_vals = tl.load(bb_row_ptr + j_idx, mask=j_mask)
        
        # Load cc[j][i] values (transpose access)
        cc_vals = tl.zeros_like(j_idx, dtype=tl.float32)
        for k in range(BLOCK_SIZE):
            j_val = j_idx[k]
            cc_val = tl.load(cc_ptr + j_val * N + i, mask=(j_val < N))
            cc_vals = tl.where(tl.arange(0, BLOCK_SIZE) == k, cc_val, cc_vals)
        
        # Compute: aa[i][j] = aa[i][j]*cc[j][i] + bb[i][j]
        result = aa_vals * cc_vals + bb_vals
        
        # Store result
        tl.store(aa_row_ptr + j_idx, result, mask=j_mask)

def s1115_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    s1115_kernel[grid](
        aa, bb, cc,
        N, BLOCK_SIZE
    )