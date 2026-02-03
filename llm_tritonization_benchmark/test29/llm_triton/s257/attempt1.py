import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Load a[i-1] (scalar, broadcast to vector)
        a_prev = tl.load(a_ptr + (i - 1))
        
        # Load aa[j][i] for all j
        aa_ji_ptrs = aa_ptr + j_offsets * len_2d + i
        aa_ji = tl.load(aa_ji_ptrs, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        # Since this is computed for all j, we take the last valid j value
        a_i_values = aa_ji - a_prev
        
        # Get the last valid value (from the highest valid j)
        if len_2d <= BLOCK_SIZE:
            # All j values are in this block, take the last one
            a_i_final = tl.max(tl.where(j_mask, a_i_values, float('-inf')))
            # Actually we need the value from j = len_2d - 1
            last_j_mask = j_offsets == (len_2d - 1)
            a_i_final = tl.sum(tl.where(last_j_mask, a_i_values, 0.0))
        else:
            # Take the last valid value
            valid_mask = j_mask
            last_valid_j = tl.max(tl.where(valid_mask, j_offsets, -1))
            last_j_mask = j_offsets == last_valid_j
            a_i_final = tl.sum(tl.where(last_j_mask, a_i_values, 0.0))
        
        # Store a[i]
        tl.store(a_ptr + i, a_i_final)
        
        # Load bb[j][i] for all j
        bb_ji_ptrs = bb_ptr + j_offsets * len_2d + i
        bb_ji = tl.load(bb_ji_ptrs, mask=j_mask)
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        aa_ji_new = a_i_final + bb_ji
        
        # Store aa[j][i] for all j
        tl.store(aa_ji_ptrs, aa_ji_new, mask=j_mask)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (1,)
    
    s257_kernel[grid](
        a, aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )