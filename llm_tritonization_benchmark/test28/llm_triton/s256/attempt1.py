import triton
import triton.language as tl
import torch

@triton.jit
def s256_kernel(a_ptr, aa_ptr, bb_ptr, d_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_mask = i_offsets < len_2d
    
    # Load bb[j][i] and d[j] for current j (broadcast d[j] to match bb shape)
    bb_vals = tl.load(bb_ptr + i_offsets, mask=i_mask)
    d_val = tl.load(d_ptr)
    
    # Load a[j-1]
    a_prev = tl.load(a_ptr)
    
    # Compute a[j] = 1.0 - a[j-1]
    a_val = 1.0 - a_prev
    
    # Store a[j]
    tl.store(a_ptr + 1, a_val)
    
    # Compute aa[j][i] = a[j] + bb[j][i] * d[j]
    aa_vals = a_val + bb_vals * d_val
    
    # Store aa[j][i]
    tl.store(aa_ptr + i_offsets, aa_vals, mask=i_mask)

def s256_triton(a, aa, bb, d, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    
    for j in range(1, len_2d):
        # Calculate pointers for current j
        a_ptr = a
        aa_ptr = aa[j]
        bb_ptr = bb[j]
        d_ptr = d[j:j+1]  # Single element as tensor
        
        # Launch kernel for all i values in parallel
        s256_kernel[(1,)](
            a_ptr + j - 1,  # Point to a[j-1] for reading
            aa_ptr,
            bb_ptr,
            d_ptr,
            len_2d,
            BLOCK_SIZE=BLOCK_SIZE
        )