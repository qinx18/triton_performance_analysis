import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, i, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Load a[i-1] from copy (scalar)
    a_i_minus_1 = tl.load(a_copy_ptr + (i - 1))
    
    # Load aa[0][i] to compute a[i]
    first_aa_i = tl.load(aa_ptr + i)
    
    # Compute a[i] = aa[0][i] - a[i-1]
    a_i = first_aa_i - a_i_minus_1
    
    # Store a[i] once
    if tl.program_id(0) == 0:
        tl.store(a_ptr + i, a_i)
    
    # Load aa[j][i] for all j values
    aa_ji_offsets = j_offsets * LEN_2D + i
    aa_ji = tl.load(aa_ptr + aa_ji_offsets, mask=j_mask)
    
    # Load bb[j][i] for all j values  
    bb_ji_offsets = j_offsets * LEN_2D + i
    bb_ji = tl.load(bb_ptr + bb_ji_offsets, mask=j_mask)
    
    # Update aa[j][i] = a[i] + bb[j][i] for all j using the scalar a[i]
    new_aa_ji = aa_ji - a_i_minus_1 + bb_ji
    
    # Store aa[j][i] for all j
    tl.store(aa_ptr + aa_ji_offsets, new_aa_ji, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    
    # Create read-only copy of array a
    a_copy = a.clone()
    
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    grid = (1,)
    
    # Sequential execution over i dimension
    for i in range(1, LEN_2D):
        s257_kernel[grid](
            a, a_copy, aa, bb,
            i, LEN_2D, BLOCK_SIZE
        )