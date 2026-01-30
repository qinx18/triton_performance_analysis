import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get j indices for this block
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar)
    a_i_minus_1 = tl.load(a_ptr + (i - 1))
    
    # Load aa[j][i] and bb[j][i] for all j values in this block
    aa_offsets = j_offsets * LEN_2D + i
    bb_offsets = j_offsets * LEN_2D + i
    
    aa_ji = tl.load(aa_ptr + aa_offsets, mask=j_mask)
    bb_ji = tl.load(bb_ptr + bb_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    a_i_values = aa_ji - a_i_minus_1
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j
    aa_ji_new = a_i_values + bb_ji
    
    # Store updated aa[j][i] values
    tl.store(aa_ptr + aa_offsets, aa_ji_new, mask=j_mask)
    
    # Store the last a[i] value (from the last valid j in the original sequential order)
    # We need to find the highest j index in this block that is valid
    block_id = tl.program_id(0)
    block_start = block_id * BLOCK_SIZE
    block_end = tl.minimum(block_start + BLOCK_SIZE, LEN_2D)
    
    # If this block contains j = LEN_2D - 1 (the last j), store a[i]
    if block_end == LEN_2D:
        last_j_in_block = LEN_2D - 1 - block_start
        if last_j_in_block >= 0 and last_j_in_block < BLOCK_SIZE:
            final_a_i = a_i_values[last_j_in_block]
            tl.store(a_ptr + i, final_a_i)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    num_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Process each i sequentially
    for i in range(1, LEN_2D):
        s257_kernel[(num_blocks,)](
            a, aa, bb, i, LEN_2D, BLOCK_SIZE
        )