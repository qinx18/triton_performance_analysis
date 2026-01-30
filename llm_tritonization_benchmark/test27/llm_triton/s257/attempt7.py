import triton
import triton.language as tl
import torch

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, len_2d, BLOCK_SIZE):
        current_j_offsets = block_start + j_offsets
        current_j_mask = current_j_offsets < len_2d
        
        # Load aa[j][i] and bb[j][i] for current block
        aa_vals = tl.load(aa_ptr + current_j_offsets, mask=current_j_mask, other=0.0)
        bb_vals = tl.load(bb_ptr + current_j_offsets, mask=current_j_mask, other=0.0)
        
        # Load current a[i]
        a_i = tl.load(a_ptr)
        
        # Store aa[j][i] = a[i] + bb[j][i]
        new_aa_vals = a_i + bb_vals
        tl.store(aa_ptr + current_j_offsets, new_aa_vals, mask=current_j_mask)

def s257_triton(a, aa, bb):
    len_2d = aa.shape[0]
    BLOCK_SIZE = 256
    
    for i in range(1, len_2d):
        # First compute new a[i] from last j iteration
        a[i] = aa[-1, i] - a[i-1]
        
        # Launch kernel to update all aa[j][i] = a[i] + bb[j][i]
        s257_kernel[(1,)](
            a[i:i+1], aa[:, i], bb[:, i], len_2d, BLOCK_SIZE=BLOCK_SIZE
        )