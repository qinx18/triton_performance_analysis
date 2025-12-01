import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(
    a_ptr,
    a_copy_ptr,
    aa_ptr,
    bb_ptr,
    LEN_2D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for i in range(1, LEN_2D):
        block_start = 0
        for j_start in range(0, LEN_2D, BLOCK_SIZE):
            j_offsets = j_start + offsets
            j_mask = j_offsets < LEN_2D
            
            # Load a[i-1] (scalar broadcast)
            a_prev = tl.load(a_copy_ptr + (i - 1))
            
            # Load aa[j][i] values
            aa_indices = j_offsets * LEN_2D + i
            aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask)
            
            # Compute a[i] = aa[j][i] - a[i-1]
            # Since all j iterations compute the same a[i], we only need one
            if j_start == 0:
                # Use first valid aa value to compute a[i]
                a_new = aa_vals[0] - a_prev
                tl.store(a_ptr + i, a_new)
            
            # Load bb[j][i] values
            bb_indices = j_offsets * LEN_2D + i
            bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask)
            
            # Load the updated a[i]
            a_new = tl.load(a_ptr + i)
            
            # Compute aa[j][i] = a[i] + bb[j][i]
            aa_new_vals = a_new + bb_vals
            
            # Store updated aa[j][i] values
            tl.store(aa_ptr + aa_indices, aa_new_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    
    # Create read-only copy for WAR dependency handling
    a_copy = a.clone()
    
    grid = (1,)
    s257_kernel[grid](
        a,
        a_copy,
        aa,
        bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )