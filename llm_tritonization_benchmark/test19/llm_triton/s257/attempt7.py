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
    
    # Compute a[i] = aa[j][i] - a[i-1] (last valid j wins)
    a_new = aa_vals - a_prev
    
    # Find the last valid j and use its computed a[i] value
    valid_mask = j_offsets < LEN_2D
    if tl.sum(valid_mask.to(tl.int32)) > 0:
        # Get the last valid index
        last_valid_idx = LEN_2D - 1 - j_base
        if last_valid_idx >= BLOCK_SIZE:
            last_valid_idx = BLOCK_SIZE - 1
        
        # Use the value from the highest valid j
        for idx in range(BLOCK_SIZE - 1, -1, -1):
            if j_base + idx < LEN_2D:
                final_a = a_new[idx] if idx < BLOCK_SIZE else a_new[BLOCK_SIZE-1]
                break
        
        # Store final a[i]
        if j_base + BLOCK_SIZE > LEN_2D:
            # This block contains the last j values
            tl.store(a_ptr + i_val, final_a)
    
    # Compute and store aa[j][i] = a[i] + bb[j][i]
    aa_new = a_new + bb_vals
    tl.store(aa_ptr + aa_offsets, aa_new, mask=mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)