import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, a_copy_ptr, aa_ptr, bb_ptr, d_ptr, j_idx, i_idx, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    current_i = i_idx + i_offsets
    mask = current_i < LEN_2D
    
    # Load a[j-1] from copy (scalar)
    a_prev = tl.load(a_copy_ptr + (j_idx - 1))
    
    # Compute a[j] = 1.0 - a[j-1] (only once per j)
    a_val = 1.0 - a_prev
    
    # Store a[j] (scalar, only for first block of i)
    if i_idx == 0:
        tl.store(a_ptr + j_idx, a_val)
    
    # Load d[j] (scalar)
    d_val = tl.load(d_ptr + j_idx)
    
    # Compute 2D indices for aa and bb
    aa_bb_offsets = j_idx * LEN_2D + current_i
    
    # Load bb[j][i] values
    bb_vals = tl.load(bb_ptr + aa_bb_offsets, mask=mask)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_val + bb_vals * d_val
    
    # Store aa[j][i] values
    tl.store(aa_ptr + aa_bb_offsets, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    # Sequential loop over j, parallel over i
    for j in range(1, LEN_2D):
        num_blocks_i = triton.cdiv(LEN_2D, BLOCK_SIZE)
        for i_block in range(num_blocks_i):
            i_start = i_block * BLOCK_SIZE
            s256_kernel[(1,)](
                a, a_copy, aa, bb, d,
                j, i_start, LEN_2D, BLOCK_SIZE
            )