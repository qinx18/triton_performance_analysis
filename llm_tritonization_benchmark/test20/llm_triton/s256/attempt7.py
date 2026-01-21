import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i_start, j, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
    mask = i_offsets < LEN_2D
    
    # Load a[j] and a[j-1]
    a_j_prev = tl.load(a_ptr + j - 1)
    a_j = 1.0 - a_j_prev
    tl.store(a_ptr + j, a_j)
    
    # Load bb[j][i] and d[j]
    bb_offsets = j * LEN_2D + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    d_j = tl.load(d_ptr + j)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_j + bb_vals * d_j
    
    # Store aa[j][i]
    aa_offsets = j * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    for j in range(1, LEN_2D):
        for i_start in range(0, LEN_2D, BLOCK_SIZE):
            s256_kernel[(1,)](
                a, aa, bb, d, i_start, j, LEN_2D, BLOCK_SIZE=BLOCK_SIZE
            )