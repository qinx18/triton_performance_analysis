import triton
import triton.language as tl

@triton.jit
def s256_kernel(aa_ptr, bb_ptr, d_ptr, a_j_val, j_val, N, BLOCK_SIZE: tl.constexpr):
    # Get thread IDs for i dimension
    i_offset = tl.program_id(0) * BLOCK_SIZE
    i_offsets = i_offset + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N
    
    # Load bb[j][i] and d[j]
    bb_offsets = j_val * N + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
    d_val = tl.load(d_ptr + j_val)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_j_val + bb_vals * d_val
    
    # Store aa[j][i]
    aa_offsets = j_val * N + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Loop over i first (outer), then j sequentially (inner)
    for i in range(N):
        for j in range(1, N):
            # Compute a[j] = 1.0 - a[j-1]
            a[j] = 1.0 - a[j-1]
            # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
            aa[j, i] = a[j] + bb[j, i] * d[j]