import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get j indices for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar broadcast)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # Load aa[j][i] for all j values
    aa_offsets = j_offsets * LEN_2D + i
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    
    # Load bb[j][i] for all j values
    bb_offsets = j_offsets * LEN_2D + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    # Since each j overwrites a[i], we compute all values
    a_vals = aa_vals - a_prev
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j
    new_aa_vals = a_vals + bb_vals
    
    # Store aa[j][i] for all j
    tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=j_mask)
    
    # Store a[i] - the last value computed will be the final result
    # We need to find the last valid j value
    last_valid_j = LEN_2D - 1
    if last_valid_j < BLOCK_SIZE:
        # Store the value from the last valid j
        final_a_val = tl.load(aa_ptr + last_valid_j * LEN_2D + i) - tl.load(bb_ptr + last_valid_j * LEN_2D + i)
        tl.store(a_ptr + i, final_a_val)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Sequential loop over i dimension
    for i in range(1, LEN_2D):
        BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
        
        s257_kernel[(1,)](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )