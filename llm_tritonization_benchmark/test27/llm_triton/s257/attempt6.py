import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < len_2d
    
    for block_start in range(0, len_2d, BLOCK_SIZE):
        current_j_offsets = block_start + j_offsets
        current_j_mask = current_j_offsets < len_2d
        
        # Load aa[j][i] and bb[j][i]
        aa_vals = tl.load(aa_ptr + current_j_offsets, mask=current_j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + current_j_offsets, mask=current_j_mask, other=0.0)
        
        # Compute a[i] + bb[j][i] for all j
        a_i_broadcast = tl.broadcast_to(tl.load(a_ptr), [BLOCK_SIZE])
        new_aa_vals = a_i_broadcast + bb_vals
        
        # Store aa[j][i] = a[i] + bb[j][i]
        tl.store(aa_ptr + current_j_offsets, new_aa_vals, mask=current_j_mask)

def s257_triton(a, aa, bb):
    len_2d = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, len_2d):
        # Compute a[i] = aa[j][i] - a[i-1] using reduction
        # First, get all aa[j][i] values
        aa_col_i = aa[:, i]  # Shape: [len_2d]
        a_i_minus_1 = a[i-1]
        
        # Compute new a[i] - use last j value (overwrite pattern)
        a[i] = aa_col_i[-1] - a_i_minus_1
        
        # Launch kernel to update aa[:,i]
        aa_ptr = aa[:, i]
        bb_ptr = bb[:, i]
        
        s257_kernel[(1,)](
            a[i:i+1], aa_ptr, bb_ptr, len_2d, BLOCK_SIZE=BLOCK_SIZE
        )