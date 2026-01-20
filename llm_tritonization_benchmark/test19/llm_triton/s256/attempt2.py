import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    # Get thread IDs for j dimension
    j_offset = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE)
    j_mask = (j_offsets < N) & (j_offsets >= 1)  # j starts from 1
    
    # Load a[j-1] values
    a_prev_offsets = j_offsets - 1
    a_prev = tl.load(a_ptr + a_prev_offsets, mask=j_mask, other=0.0)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_vals = 1.0 - a_prev
    
    # Store a[j]
    tl.store(a_ptr + j_offsets, a_vals, mask=j_mask)
    
    # Load bb[j][i] and d[j] for aa computation
    bb_offsets = j_offsets * N + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    d_vals = tl.load(d_ptr + j_offsets, mask=j_mask, other=0.0)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_vals + bb_vals * d_vals
    
    # Store aa[j][i]
    aa_offsets = j_offsets * N + i_val
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=j_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Loop over i sequentially, parallelize j
    for i in range(N):
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s256_kernel[grid](
            a, aa, bb, d, 
            i, N, 
            BLOCK_SIZE=BLOCK_SIZE
        )