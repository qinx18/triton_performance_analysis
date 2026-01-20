import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_base = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_base + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Load a[i-1] once
    a_prev = tl.load(a_ptr + (i_val - 1))
    
    # Process all j values in this block
    for j_idx in range(BLOCK_SIZE):
        j = j_base + j_idx
        if j < LEN_2D:
            # Load aa[j][i] and bb[j][i]
            aa_offset = j * LEN_2D + i_val
            bb_offset = j * LEN_2D + i_val
            
            aa_val = tl.load(aa_ptr + aa_offset)
            bb_val = tl.load(bb_ptr + bb_offset)
            
            # Compute a[i] = aa[j][i] - a[i-1]
            a_new = aa_val - a_prev
            
            # Store a[i] (overwrites for each j)
            tl.store(a_ptr + i_val, a_new)
            
            # Compute and store aa[j][i] = a[i] + bb[j][i]
            aa_new = a_new + bb_val
            tl.store(aa_ptr + aa_offset, aa_new)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    for i in range(1, LEN_2D):
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)