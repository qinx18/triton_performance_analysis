import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, N, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < N
    
    # Load a[i] and a[i-1]
    a_i_prev = tl.load(a_ptr + (i - 1))
    
    # For each j, compute a[i] = aa[j][i] - a[i-1]
    # Since all j iterations overwrite a[i], we need the last valid j value
    # Load all aa[j][i] values
    aa_offsets = j_offsets * N + i
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + aa_offsets, mask=mask)
    
    # Compute new a[i] for each j
    new_a_vals = aa_vals - a_i_prev
    
    # Since all j iterations overwrite a[i], use the last valid j value
    valid_js = tl.where(mask, j_offsets, -1)
    last_valid_j = tl.max(valid_js)
    
    # Get the a[i] value from the last j iteration
    final_a_i = tl.where(j_offsets == last_valid_j, new_a_vals, 0.0)
    final_a_i = tl.sum(final_a_i)
    
    # Store the final a[i]
    tl.store(a_ptr + i, final_a_i)
    
    # Update all aa[j][i] = a[i] + bb[j][i]
    new_aa_vals = final_a_i + bb_vals
    tl.store(aa_ptr + aa_offsets, new_aa_vals, mask=mask)

def s257_triton(a, aa, bb):
    N = aa.shape[0]  # LEN_2D
    BLOCK_SIZE = 256
    
    # Sequential loop over i from 1 to N-1
    for i in range(1, N):
        # Launch kernel for all j values in parallel
        grid = (triton.cdiv(N, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, N, BLOCK_SIZE=BLOCK_SIZE)