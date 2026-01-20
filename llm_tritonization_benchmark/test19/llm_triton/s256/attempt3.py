import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, i_val, N, BLOCK_SIZE: tl.constexpr):
    # Get thread IDs for j dimension
    j_offset = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_offset + tl.arange(0, BLOCK_SIZE)
    j_mask = (j_offsets < N) & (j_offsets >= 1)  # j starts from 1
    
    # For each j, compute a[j] = 1.0 - a[j-1]
    # First load a[j-1]
    a_prev_offsets = j_offsets - 1
    a_prev_mask = a_prev_offsets >= 0
    a_prev = tl.load(a_ptr + a_prev_offsets, mask=a_prev_mask, other=0.0)
    
    # Compute new a[j] = 1.0 - a[j-1]
    a_new = 1.0 - a_prev
    
    # Store a[j]
    tl.store(a_ptr + j_offsets, a_new, mask=j_mask)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    # Load bb[j][i] and d[j]
    bb_offsets = j_offsets * N + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=j_mask, other=0.0)
    d_vals = tl.load(d_ptr + j_offsets, mask=j_mask, other=0.0)
    
    # Compute aa[j][i]
    aa_vals = a_new + bb_vals * d_vals
    
    # Store aa[j][i]
    aa_offsets = j_offsets * N + i_val
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=j_mask)

def s256_triton(a, aa, bb, d):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Loop over j sequentially (j from 1 to N-1), then over i in parallel
    for j in range(1, N):
        # Compute a[j] = 1.0 - a[j-1]
        a[j] = 1.0 - a[j-1]
        
        # Parallelize over i
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s256_i_kernel[grid](
            a, aa, bb, d,
            j, N,
            BLOCK_SIZE=BLOCK_SIZE
        )

@triton.jit
def s256_i_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, j_val, N, BLOCK_SIZE: tl.constexpr):
    # Get thread IDs for i dimension
    i_offset = tl.program_id(0) * BLOCK_SIZE
    i_offsets = i_offset + tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N
    
    # Load a[j] (scalar broadcast)
    a_val = tl.load(a_ptr + j_val)
    
    # Load bb[j][i] and d[j]
    bb_offsets = j_val * N + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask, other=0.0)
    d_val = tl.load(d_ptr + j_val)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_val + bb_vals * d_val
    
    # Store aa[j][i]
    aa_offsets = j_val * N + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)