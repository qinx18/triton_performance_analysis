import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr, j_idx, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < LEN_2D
    
    # Load a[j-1] from copy (scalar broadcast)
    a_prev = tl.load(a_copy_ptr + (j_idx - 1))
    
    # Compute a[j] = 1.0 - a[j-1]
    a_val = 1.0 - a_prev
    
    # Store a[j] (scalar)
    tl.store(a_ptr + j_idx, a_val)
    
    # Load d[j] (scalar broadcast)
    d_val = tl.load(d_ptr + j_idx)
    
    # Compute 2D indices for aa and bb
    aa_bb_offsets = j_idx * LEN_2D + i_offsets
    
    # Load bb[j][i] values
    bb_vals = tl.load(bb_ptr + aa_bb_offsets, mask=i_mask)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_val + bb_vals * d_val
    
    # Store aa[j][i] values
    tl.store(aa_ptr + aa_bb_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Sequential loop over j, parallel over i
    for j in range(1, LEN_2D):
        s256_kernel[(1,)](
            a, a_copy, aa, bb, d,
            j, LEN_2D, BLOCK_SIZE
        )