import torch
import triton
import triton.language as tl

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, j_val, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < LEN_2D
    
    # Load a[j] and a[j-1]
    a_j = tl.load(a_ptr + j_val)
    a_j_minus_1 = tl.load(a_ptr + j_val - 1)
    
    # Compute new a[j] value
    new_a_j = 1.0 - a_j_minus_1
    
    # Store new a[j]
    tl.store(a_ptr + j_val, new_a_j)
    
    # Load d[j]
    d_j = tl.load(d_ptr + j_val)
    
    # Load bb[j][i] values for all i
    bb_offsets = j_val * LEN_2D + i_offsets
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=i_mask)
    
    # Compute aa[j][i] = new_a_j + bb[j][i] * d[j]
    aa_vals = new_a_j + bb_vals * d_j
    
    # Store aa[j][i] values
    aa_offsets = j_val * LEN_2D + i_offsets
    tl.store(aa_ptr + aa_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(LEN_2D)
    
    for j in range(1, LEN_2D):
        s256_kernel[(1,)](
            a, aa, bb, d, j, LEN_2D, BLOCK_SIZE
        )