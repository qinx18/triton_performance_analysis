import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Process all j values in parallel
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar value)
    a_i_minus_1 = tl.load(a_ptr + (i - 1))
    
    # Load aa[j][i] for all j values
    aa_ji_offsets = j_offsets * LEN_2D + i
    aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask)
    
    # Load bb[j][i] for all j values  
    bb_ji_offsets = j_offsets * LEN_2D + i
    bb_ji = tl.load(bb_ptr + bb_ji_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    a_i = aa_ji - a_i_minus_1
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j
    aa_ji_new = a_i + bb_ji
    
    # Store aa[j][i] for all j values
    tl.store(aa_ptr + aa_ji_offsets, aa_ji_new, mask=j_mask)
    
    # For a[i], the last j overwrites previous values, so use the last valid j
    last_valid_j = LEN_2D - 1
    if last_valid_j < BLOCK_SIZE:
        last_aa_ji = tl.load(aa_ptr + last_valid_j * LEN_2D + i) - a_i_minus_1
        # Only one thread should write to a[i]
        if tl.program_id(0) == 0:
            tl.store(a_ptr + i, last_aa_ji)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Block size for j dimension
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    # Launch kernel sequentially for each i value
    for i in range(1, LEN_2D):
        s257_kernel[(1,)](
            a, aa, bb, i,
            LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE
        )