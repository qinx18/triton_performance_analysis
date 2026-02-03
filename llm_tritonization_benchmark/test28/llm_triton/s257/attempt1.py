import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Load a[i-1] (scalar broadcast)
        a_prev = tl.load(a_ptr + (i - 1))
        
        # Load aa[j][i] for all j
        aa_offsets = j_offsets * len_2d + i
        aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
        
        # Compute a[i] = aa[j][i] - a[i-1]
        # Since this is overwritten for each j, we take the last valid computation
        a_new = aa_vals - a_prev
        
        # Load bb[j][i] for all j
        bb_offsets = j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
        
        # Compute aa[j][i] = a[i] + bb[j][i]
        aa_new = a_new + bb_vals
        
        # Store aa[j][i]
        tl.store(aa_ptr + aa_offsets, aa_new, mask=j_mask)
        
        # Store a[i] (using the last computed value)
        # We need to extract a scalar from the vector for the final store
        # Use the last valid j value
        valid_mask = j_mask
        if tl.sum(valid_mask.to(tl.int32)) > 0:
            # Find the last valid index
            last_valid_j = len_2d - 1 if len_2d <= BLOCK_SIZE else BLOCK_SIZE - 1
            a_final = tl.load(aa_ptr + last_valid_j * len_2d + i) - bb_vals
            a_final_scalar = a_final + bb_vals - bb_vals  # Get just aa value
            a_final_scalar = aa_vals - a_prev  # Recompute to get the right value
            
            # Actually, we need the value from the last j iteration
            # Let's store using the last element that was computed
            last_j_offset = (len_2d - 1) * len_2d + i if len_2d <= BLOCK_SIZE else (BLOCK_SIZE - 1) * len_2d + i
            last_aa = tl.load(aa_ptr + last_j_offset)
            last_bb = tl.load(bb_ptr + last_j_offset)
            a_final_val = last_aa - last_bb
            
            tl.store(a_ptr + i, a_final_val)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (1,)
    
    s257_kernel[grid](
        a, aa, bb, 
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )