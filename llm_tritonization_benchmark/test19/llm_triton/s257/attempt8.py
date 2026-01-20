import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_base = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_base + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Load a[i-1] once
    a_prev = tl.load(a_ptr + (i_val - 1))
    
    # Load aa[j][i] for all valid j in this block
    aa_offsets = j_offsets * LEN_2D + i_val
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Load bb[j][i] for all valid j in this block
    bb_offsets = j_offsets * LEN_2D + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    a_new_vals = aa_vals - a_prev
    
    # Find the last valid j and store its value to a[i]
    # We need to find the maximum valid j in this block
    last_valid_j = -1
    last_a_val = a_prev  # fallback value
    
    # Check each position to find the last valid one
    for k in range(BLOCK_SIZE):
        j_idx = j_base + k
        if j_idx < LEN_2D:
            last_valid_j = k
            last_a_val = a_new_vals[k] if k < BLOCK_SIZE else last_a_val
    
    # Store a[i] if this block contains valid indices
    if last_valid_j >= 0:
        if j_base + BLOCK_SIZE >= LEN_2D:
            # This is the last block, store the final value
            tl.store(a_ptr + i_val, last_a_val)
    
    # Compute and store aa[j][i] = a[i] + bb[j][i]
    # Use the computed a_new_vals for each j
    aa_new = a_new_vals + bb_vals
    tl.store(aa_ptr + aa_offsets, aa_new, mask=mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)