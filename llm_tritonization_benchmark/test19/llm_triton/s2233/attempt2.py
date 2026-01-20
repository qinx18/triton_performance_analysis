import triton
import triton.language as tl
import torch

@triton.jit
def s2233_kernel(aa_ptr, bb_ptr, cc_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_idx = pid * BLOCK_SIZE + i_offsets
    i_mask = i_idx < N
    
    # Only process valid i indices (starting from 1)
    valid_i_mask = (i_idx >= 1) & i_mask
    
    for j in range(1, N):
        # First loop: aa[j][i] = aa[j-1][i] + cc[j][i]
        aa_read_offsets = (j - 1) * N + i_idx
        aa_write_offsets = j * N + i_idx
        cc_read_offsets = j * N + i_idx
        
        aa_prev = tl.load(aa_ptr + aa_read_offsets, mask=valid_i_mask, other=0.0)
        cc_val = tl.load(cc_ptr + cc_read_offsets, mask=valid_i_mask, other=0.0)
        aa_new = aa_prev + cc_val
        tl.store(aa_ptr + aa_write_offsets, aa_new, mask=valid_i_mask)
        
        # Second loop: bb[i][j] = bb[i-1][j] + cc[i][j]
        bb_read_offsets = (i_idx - 1) * N + j
        bb_write_offsets = i_idx * N + j
        cc_read_offsets_2 = i_idx * N + j
        
        bb_prev = tl.load(bb_ptr + bb_read_offsets, mask=valid_i_mask, other=0.0)
        cc_val_2 = tl.load(cc_ptr + cc_read_offsets_2, mask=valid_i_mask, other=0.0)
        bb_new = bb_prev + cc_val_2
        tl.store(bb_ptr + bb_write_offsets, bb_new, mask=valid_i_mask)

def s2233_triton(aa, bb, cc):
    N = aa.shape[0]
    BLOCK_SIZE = min(64, triton.next_power_of_2(N))
    
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s2233_kernel[grid](aa, bb, cc, N, BLOCK_SIZE=BLOCK_SIZE)