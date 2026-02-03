import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for i in range(1, len_2d):
        # Load a[i-1] (scalar)
        a_prev = tl.load(a_ptr + (i - 1))
        
        # For each j, compute a[i] = aa[j][i] - a[i-1]
        # Since all j values overwrite the same a[i], the last j wins
        for j in range(len_2d):
            # Load aa[j][i]
            aa_j_i = tl.load(aa_ptr + j * len_2d + i)
            # Compute a[i] = aa[j][i] - a[i-1]
            a_i = aa_j_i - a_prev
            # Store a[i]
            tl.store(a_ptr + i, a_i)
        
        # Now load the final a[i] value and update all aa[j][i] in parallel
        final_a_i = tl.load(a_ptr + i)
        
        # Load bb[j][i] for all j in parallel
        bb_ptrs = bb_ptr + j_offsets * len_2d + i
        bb_vals = tl.load(bb_ptrs, mask=j_mask, other=0.0)
        
        # Compute aa[j][i] = a[i] + bb[j][i] for all j in parallel
        aa_new_vals = final_a_i + bb_vals
        
        # Store aa[j][i] for all j in parallel
        aa_ptrs = aa_ptr + j_offsets * len_2d + i
        tl.store(aa_ptrs, aa_new_vals, mask=j_mask)

def s257_triton(a, aa, bb, len_2d):
    BLOCK_SIZE = triton.next_power_of_2(len_2d)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    
    grid = (1,)
    
    s257_kernel[grid](
        a, aa, bb, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )