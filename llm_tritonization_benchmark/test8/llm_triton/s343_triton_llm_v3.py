import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, LEN_2D):
    k = -1
    
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            # Load bb[j][i]
            bb_offset = j * LEN_2D + i
            bb_val = tl.load(bb_ptr + bb_offset)
            
            if bb_val > 0.0:
                k += 1
                # Load aa[j][i]
                aa_offset = j * LEN_2D + i
                aa_val = tl.load(aa_ptr + aa_offset)
                # Store to flat_2d_array[k]
                tl.store(flat_2d_array_ptr + k, aa_val)

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    
    grid = (1,)
    s343_kernel[grid](
        aa, bb, flat_2d_array,
        LEN_2D
    )