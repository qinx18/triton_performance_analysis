import torch
import triton
import triton.language as tl

@triton.jit
def s257_kernel(a_ptr, aa_ptr, bb_ptr, i_val, LEN_2D, BLOCK_SIZE: tl.constexpr):
    j_offsets = tl.arange(0, BLOCK_SIZE)
    mask = j_offsets < LEN_2D
    
    # Load a[i-1] (scalar broadcast)
    a_prev = tl.load(a_ptr + (i_val - 1))
    
    # Process all j values sequentially to maintain correct dependency
    current_a = a_prev
    for j in range(LEN_2D):
        # Load aa[j][i]
        aa_offset = j * LEN_2D + i_val
        aa_val = tl.load(aa_ptr + aa_offset)
        
        # Load bb[j][i] 
        bb_offset = j * LEN_2D + i_val
        bb_val = tl.load(bb_ptr + bb_offset)
        
        # a[i] = aa[j][i] - a[i-1]
        current_a = aa_val - current_a
        
        # aa[j][i] = a[i] + bb[j][i]
        new_aa = current_a + bb_val
        tl.store(aa_ptr + aa_offset, new_aa)
    
    # Store final a[i]
    tl.store(a_ptr + i_val, current_a)

def s257_triton(a, aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 1
    
    for i in range(1, LEN_2D):
        grid = (1,)
        s257_kernel[grid](a, aa, bb, i, LEN_2D, BLOCK_SIZE=BLOCK_SIZE)