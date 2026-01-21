import triton
import triton.language as tl
import torch

@triton.jit
def s1119_kernel(aa, bb, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N

    for i in range(1, N):
        # Load aa[i-1][j]
        aa_prev_ptrs = aa + (i-1) * N + j_idx
        aa_prev = tl.load(aa_prev_ptrs, mask=j_mask, other=0.0)
        
        # Load bb[i][j]
        bb_ptrs = bb + i * N + j_idx
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Compute and store aa[i][j] = aa[i-1][j] + bb[i][j]
        result = aa_prev + bb_vals
        aa_ptrs = aa + i * N + j_idx
        tl.store(aa_ptrs, result, mask=j_mask)
        
        # Add synchronization to ensure all threads complete before next iteration
        tl.debug_barrier()

def s1119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s1119_kernel[grid](aa, bb, N, BLOCK_SIZE=BLOCK_SIZE)