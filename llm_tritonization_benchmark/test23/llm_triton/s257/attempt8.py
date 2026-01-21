import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < N
    
    # Load a[i-1] once
    a_i_prev = tl.load(a_ptr + (i - 1))
    
    # Load aa[j][i] values for all j
    aa_offsets = j_offsets * N + i
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    
    # Compute a[i] = aa[j][i] - a[i-1] for all j
    new_a_vals = aa_vals - a_i_prev
    
    # The last valid j overwrites a[i], so find the maximum valid j
    last_valid_j = N - 1
    a_final = tl.load(aa_ptr + last_valid_j * N + i) - a_i_prev
    
    # Store final a[i]
    tl.store(a_ptr + i, a_final)
    
    # Now compute aa[j][i] = a[i] + bb[j][i] for all j
    bb_vals = tl.load(bb_ptr + aa_offsets, mask=mask)
    new_aa_vals = a_final + bb_vals
    
    # Store updated aa[j][i] values
    tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=mask)

def s257_triton(a, aa, bb):
    N = aa.shape[0]  # LEN_2D
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    # Sequential loop over i from 1 to N-1
    for i in range(1, N):
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)