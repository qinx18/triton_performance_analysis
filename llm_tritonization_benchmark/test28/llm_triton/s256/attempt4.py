import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(aa_ptr, bb_ptr, d_ptr, a_val, i_idx, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < (len_2d - 1)
    j_indices = j_offsets + 1
    
    # Calculate 2D indices for aa[j][i] and bb[j][i]
    aa_indices = j_indices * len_2d + i_idx
    bb_indices = j_indices * len_2d + i_idx
    
    # Load bb[j][i] and d[j]
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask)
    d_vals = tl.load(d_ptr + j_indices, mask=j_mask)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_val + bb_vals * d_vals
    
    # Store aa[j][i]
    tl.store(aa_ptr + aa_indices, aa_vals, mask=j_mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    for i in range(len_2d):
        for j in range(1, len_2d):
            # Compute a[j] = 1.0 - a[j-1]
            a[j] = 1.0 - a[j-1]
        
        # Launch kernel for all j values for this i
        s256_kernel[(1,)](
            aa,
            bb,
            d,
            a[len_2d-1].item(),
            i,
            len_2d,
            BLOCK_SIZE=BLOCK_SIZE
        )