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
    
    # Find the last valid j value using reduction
    valid_mask = mask
    num_valid = tl.sum(valid_mask.to(tl.int32))
    
    # Get the last valid a_new_val
    last_a_val = a_prev
    if num_valid > 0:
        # Find last valid index
        last_idx = num_valid - 1
        # Extract last valid value
        last_a_val = tl.where(tl.arange(0, BLOCK_SIZE) == last_idx, a_new_vals, last_a_val)
        last_a_val = tl.sum(last_a_val)
    
    # Store a[i] if this is the last block with valid data
    is_last_block = (j_base + BLOCK_SIZE >= LEN_2D) & (num_valid > 0)
    if is_last_block:
        tl.store(a_ptr + i_val, last_a_val)
    
    # Compute and store aa[j][i] = a[i] + bb[j][i]
    aa_new = a_new_vals + bb_vals
    tl.store(aa_ptr + aa_offsets, aa_new, mask=mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)