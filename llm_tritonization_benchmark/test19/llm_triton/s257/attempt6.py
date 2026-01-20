import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get j block
    j_base = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_base + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Load a[i-1]
    a_prev = tl.load(a_ptr + (i_val - 1))
    
    # For each valid j in this block
    for j_idx in range(BLOCK_SIZE):
        j_actual = j_base + j_idx
        if j_actual < LEN_2D:
            # Load aa[j][i]
            aa_offset = j_actual * LEN_2D + i_val
            aa_val = tl.load(aa_ptr + aa_offset)
            
            # Compute a[i] = aa[j][i] - a[i-1]
            a_i = aa_val - a_prev
            
            # Store a[i]
            tl.store(a_ptr + i_val, a_i)
            
            # Load bb[j][i]
            bb_offset = j_actual * LEN_2D + i_val
            bb_val = tl.load(bb_ptr + bb_offset)
            
            # Compute and store aa[j][i] = a[i] + bb[j][i]
            aa_new = a_i + bb_val
            tl.store(aa_ptr + aa_offset, aa_new)
            
            # Update a_prev for next iteration
            a_prev = a_i

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)