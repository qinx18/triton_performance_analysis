import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE) + 1
    mask = j_offsets < LEN_2D
    
    # Load a[j-1] values
    a_prev_offsets = j_offsets - 1
    a_prev = tl.load(a_ptr + a_prev_offsets, mask=mask)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_new = 1.0 - a_prev
    
    # Store a[j]
    tl.store(a_ptr + j_offsets, a_new, mask=mask)
    
    # Load bb[j][i] and d[j] values
    bb_offsets = j_offsets * LEN_2D + i
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    d_vals = tl.load(d_ptr + j_offsets, mask=mask)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_new + bb_vals * d_vals
    
    # Store aa[j][i]
    aa_offsets = j_offsets * LEN_2D + i
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential over j, parallel over i
    for j_start in range(1, LEN_2D, BLOCK_SIZE - 1):
        j_end = min(j_start + BLOCK_SIZE - 1, LEN_2D)
        actual_block_size = j_end - j_start
        
        # Round up to nearest power of 2 for block size
        block_size_pow2 = 1
        while block_size_pow2 < actual_block_size + 1:
            block_size_pow2 *= 2
        
        for i in range(LEN_2D):
            s256_kernel[(1,)](
                a, aa, bb, d, i, LEN_2D, BLOCK_SIZE=block_size_pow2
            )