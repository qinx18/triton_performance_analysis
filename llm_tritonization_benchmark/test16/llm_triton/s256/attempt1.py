import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, N_2D, j_val, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < N_2D
    
    # Load a[j] and a[j-1]
    a_j = tl.load(a_ptr + j_val, mask=j_val < N_2D)
    a_j_minus_1 = tl.load(a_ptr + (j_val - 1), mask=j_val > 0)
    
    # Compute a[j] = 1.0 - a[j-1]
    new_a_j = 1.0 - a_j_minus_1
    
    # Store new value to a[j]
    tl.store(a_ptr + j_val, new_a_j, mask=j_val < N_2D)
    
    # Load bb[j][i] and d[j] for all i values
    bb_offsets = j_val * N_2D + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
    d_j = tl.load(d_ptr + j_val, mask=j_val < N_2D)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j] for all i
    aa_vals = new_a_j + bb_vals * d_j
    
    # Store to aa[j][i] for all i
    aa_offsets = j_val * N_2D + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    N_2D = aa.shape[0]
    
    BLOCK_SIZE = triton.next_power_of_2(N_2D)
    
    for j in range(1, N_2D):
        s256_kernel[(1,)](
            a, aa, bb, d,
            N_2D, j,
            BLOCK_SIZE=BLOCK_SIZE
        )