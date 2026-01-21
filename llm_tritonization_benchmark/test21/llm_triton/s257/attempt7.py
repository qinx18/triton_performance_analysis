import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < LEN_2D
    
    # Calculate 2D indices for aa[j][i] and bb[j][i]
    aa_indices = j_offsets * LEN_2D + i
    bb_indices = j_offsets * LEN_2D + i
    
    # Load values
    aa_vals = tl.load(aa_ptr + aa_indices, mask=j_mask, other=0.0)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=j_mask, other=0.0)
    a_prev = tl.load(a_ptr + (i - 1))
    
    # Compute a[i] = aa[j][i] - a[i-1] for each j
    a_new_vals = aa_vals - a_prev
    
    # Compute aa[j][i] = a[i] + bb[j][i]
    aa_new_vals = a_new_vals + bb_vals
    
    # Store results
    tl.store(aa_ptr + aa_indices, aa_new_vals, mask=j_mask)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Sequential loop over i
    for i in range(1, LEN_2D):
        # Launch kernel to process all j values in parallel
        grid = (triton.cdiv(LEN_2D, BLOCK_SIZE),)
        s257_kernel[grid](a, aa, bb, i, LEN_2D=LEN_2D, BLOCK_SIZE=BLOCK_SIZE)
        
        # Update a[i] with the result from the last j iteration
        # Since the loop executes j from 0 to LEN_2D-1, the last j is LEN_2D-1
        # a[i] = aa[j][i] - a[i-1], and since all j write to a[i], we need the final computation
        # The final a[i] should be aa[LEN_2D-1][i] - a[i-1]
        # But aa[LEN_2D-1][i] was updated to a[i] + bb[LEN_2D-1][i]
        # So the original aa[LEN_2D-1][i] value was used in the computation
        # We need to extract a[i] from the final computation
        last_j = LEN_2D - 1
        # The kernel computed: a[i] = aa[last_j][i] - a[i-1] (original aa value)
        # Then: aa[last_j][i] = a[i] + bb[last_j][i] (updated aa value)
        # So: a[i] = updated_aa[last_j][i] - bb[last_j][i]
        a[i] = aa[last_j, i] - bb[last_j, i]