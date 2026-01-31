import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(aa_ptr, bb_ptr, d_ptr, a_val, len_2d, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < len_2d
    
    # Load bb[j][i] and d[j]
    bb_vals = tl.load(bb_ptr + i_offsets, mask=i_mask)
    d_val = tl.load(d_ptr)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_val + bb_vals * d_val
    
    # Store aa[j][i]
    tl.store(aa_ptr + i_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    for j in range(1, len_2d):
        # Compute a[j] = 1.0 - a[j-1]
        a[j] = 1.0 - a[j-1]
        
        # Launch kernel for aa[j][i] computation
        s256_kernel[(1,)](
            aa[j],
            bb[j],
            d + j,
            a[j].item(),
            len_2d,
            BLOCK_SIZE=BLOCK_SIZE
        )