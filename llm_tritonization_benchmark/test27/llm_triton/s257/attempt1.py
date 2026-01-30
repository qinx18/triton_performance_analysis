import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    # Get current i value (passed as program ID)
    i = tl.program_id(0) + 1
    
    if i >= LEN_2D:
        return
    
    # Process all j values in parallel
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar value, broadcast to vector)
    a_i_minus_1 = tl.load(a_ptr + (i - 1))
    
    # Load aa[j][i] for all j values
    aa_ji_offsets = j_offsets * LEN_2D + i
    aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask)
    
    # Load bb[j][i] for all j values  
    bb_ji_offsets = j_offsets * LEN_2D + i
    bb_ji = tl.load(bb_ptr + bb_ji_offsets, mask=j_mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j (overwrite pattern - last j wins)
    a_i = aa_ji - a_i_minus_1
    
    # Compute aa[j][i] = a[i] + bb[j][i] for all j
    aa_ji_new = a_i + bb_ji
    
    # Store results
    # For a[i], we need the last value (from j = LEN_2D-1)
    if LEN_2D > 0:
        last_j = LEN_2D - 1
        if last_j < BLOCK_SIZE:
            a_i_final = tl.load(aa_ptr + last_j * LEN_2D + i) - a_i_minus_1
            tl.store(a_ptr + i, a_i_final)
    
    # Store aa[j][i] for all j values
    tl.store(aa_ptr + aa_ji_offsets, aa_ji_new, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Block size for j dimension
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    # Launch kernel sequentially for each i value
    for i in range(1, LEN_2D):
        s257_kernel[(1,)](
            a, aa, bb,
            LEN_2D,
            BLOCK_SIZE=BLOCK_SIZE
        )