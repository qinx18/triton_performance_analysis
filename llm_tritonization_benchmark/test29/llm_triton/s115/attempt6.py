import triton
import triton.language as tl
import torch

@triton.jit
def s115_kernel(a_ptr, aa_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    for j in range(len_2d):
        # Check if any i values in this block are valid for this j
        i_mask = (i_offsets < len_2d) & (i_offsets > j)
        
        if tl.sum(i_mask.to(tl.int32)) > 0:
            # Load a[j] (scalar broadcast)
            a_j = tl.load(a_ptr + j)
            
            # Load aa[j][i] for valid i values
            aa_ji_ptrs = aa_ptr + j * len_2d + i_offsets
            aa_ji = tl.load(aa_ji_ptrs, mask=i_mask, other=0.0)
            
            # Load a[i] for valid i values
            a_i = tl.load(a_ptr + i_offsets, mask=i_mask, other=0.0)
            
            # Compute a[i] -= aa[j][i] * a[j]
            result = a_i - aa_ji * a_j
            
            # Store back to a[i]
            tl.store(a_ptr + i_offsets, result, mask=i_mask)

def s115_triton(a, aa, len_2d):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d, BLOCK_SIZE),)
    
    s115_kernel[grid](
        a, aa, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )