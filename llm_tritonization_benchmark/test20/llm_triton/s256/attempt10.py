import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = 1 + tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Load a[j-1] values
    a_prev_vals = tl.load(a_ptr + j_offsets - 1, mask=mask)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_vals = 1.0 - a_prev_vals
    
    # Store a[j]
    tl.store(a_ptr + j_offsets, a_vals, mask=mask)
    
    # Load bb[j][i] and d[j]
    bb_offsets = j_offsets * LEN_2D + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    d_vals = tl.load(d_ptr + j_offsets, mask=mask)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_vals + bb_vals * d_vals
    
    # Store aa[j][i]
    aa_offsets = j_offsets * LEN_2D + i
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(LEN_2D):
        num_blocks = triton.cdiv(LEN_2D - 1, BLOCK_SIZE)
        if num_blocks > 0:
            s256_kernel[(num_blocks,)](
                a, aa, bb, d, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE
            )