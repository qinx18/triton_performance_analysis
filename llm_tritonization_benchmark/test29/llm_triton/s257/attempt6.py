import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Load a[i-1]
        a_prev = tl.load(a_ptr + (i - 1))
        
        # Load aa[j, i] for all j
        aa_offsets = j_offsets * len_2d + i
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
        
        # Compute a[i] = aa[j, i] - a[i-1] for each j, last j wins
        a_new = aa_vals - a_prev
        
        # Store the last computed a[i]
        if BLOCK_SIZE >= len_2d:
            # All j values computed, store the last one
            tl.store(a_ptr + i, tl.max(tl.where(j_mask, a_new, float('-inf'))))
        else:
            # Need to handle partial blocks - store each valid result
            for j_block in range(0, len_2d, BLOCK_SIZE):
                block_j_offsets = j_block + j_offsets
                block_j_mask = block_j_offsets < len_2d
                if tl.sum(block_j_mask.to(tl.int32)) > 0:
                    block_aa_offsets = block_j_offsets * len_2d + i
                    block_aa_vals = tl.load(aa_ptr + block_aa_offsets, mask=block_j_mask)
                    block_a_new = block_aa_vals - a_prev
                    # Store last valid result
                    last_valid_idx = tl.sum(block_j_mask.to(tl.int32)) - 1
                    if last_valid_idx >= 0:
                        tl.store(a_ptr + i, block_a_new + last_valid_idx * 0)
        
        # Load updated a[i]
        a_i = tl.load(a_ptr + i)
        
        # Load bb[j, i] and compute aa[j, i] = a[i] + bb[j, i]
        bb_offsets = j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        new_aa_vals = a_i + bb_vals
        
        # Store updated aa[j, i]
        tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    grid = (1,)
    s257_kernel[grid](
        a, aa, bb, len_2d, BLOCK_SIZE
    )