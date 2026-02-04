import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_base = pid * BLOCK_SIZE
    
    for j in range(len_2d):
        i_idx = i_base + i_offsets
        
        # Only process i values where i > j
        i_mask = (i_idx < len_2d) & (i_idx > j)
        
        # Load a[j] - single scalar value
        a_j = tl.load(a_ptr + j)
        
        # Load aa[j][i] for valid i indices
        aa_ptrs = aa_ptr + j * len_2d + i_idx
        aa_vals = tl.load(aa_ptrs, mask=i_mask, other=0.0)
        
        # Load a[i] for valid i indices  
        a_vals = tl.load(a_ptr + i_idx, mask=i_mask, other=0.0)
        
        # Compute: a[i] -= aa[j][i] * a[j]
        new_a_vals = a_vals - aa_vals * a_j
        
        # Store back to a[i]
        tl.store(a_ptr + i_idx, new_a_vals, mask=i_mask)

def s115_triton(a, aa, len_2d):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s115_kernel[grid](
        a, aa, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )