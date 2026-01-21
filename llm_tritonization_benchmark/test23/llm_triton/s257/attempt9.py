import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < N
    
    # Load a[i-1] once
    a_i_prev = tl.load(a_ptr + (i - 1))
    
    # Process all j values, but need to compute a[i] from the LAST j (j = N-1)
    # Load aa[N-1][i] to compute final a[i]
    final_j = N - 1
    aa_final_offset = final_j * N + i
    aa_final_val = tl.load(aa_ptr + aa_final_offset)
    a_i_new = aa_final_val - a_i_prev
    
    # Store the new a[i]
    tl.store(a_ptr + i, a_i_new)
    
    # Now compute aa[j][i] = a[i] + bb[j][i] for all j
    aa_offsets = j_offsets * N + i
    bb_vals = tl.load(bb_ptr + aa_offsets, mask=mask)
    new_aa_vals = a_i_new + bb_vals
    
    # Store updated aa[j][i] values
    tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=mask)

def s257_triton(a, aa, bb):
    N = aa.shape[0]  # LEN_2D
    BLOCK_SIZE = triton.next_power_of_2(N)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    # Sequential loop over i from 1 to N-1
    for i in range(1, N):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)