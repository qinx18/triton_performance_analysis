import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Calculate 2D indices for aa[j][i] and bb[j][i]
    aa_indices = j_offsets * LEN_2D + i
    bb_indices = j_offsets * LEN_2D + i
    
    # Load values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # For each j: a[i] = aa[j][i] - a[i-1]
    # Since all j values write to the same a[i], the last j wins
    # But we need to update aa[j][i] for each j with the corresponding a[i]
    a_new_vals = aa_vals - a_prev
    
    # Store the last a[i] value (from the highest valid j)
    valid_j_count = tl.sum(j_mask.to(tl.int32))
    if valid_j_count > 0:
        last_valid_idx = valid_j_count - 1
        last_a_val = tl.load(aa_ptr + last_valid_idx * LEN_2D + i) - a_prev
        tl.store(a_ptr + i, last_a_val)
    
    # Update aa[j][i] = a[i] + bb[j][i] for each j
    aa_new_vals = a_new_vals + bb_vals
    tl.store(aa_ptr + aa_indices, aa_new_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)