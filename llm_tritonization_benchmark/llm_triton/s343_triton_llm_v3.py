import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d: tl.constexpr):
    # This kernel processes the entire 2D array sequentially
    # since the packing operation has dependencies on the order
    
    k = -1
    for i in range(len_2d):
        for j in range(len_2d):
            bb_val = tl.load(bb_ptr + j * len_2d + i)
            if bb_val > 0.0:
                k += 1
                aa_val = tl.load(aa_ptr + j * len_2d + i)
                tl.store(flat_2d_array_ptr + k, aa_val)

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Reset the flat array
    flat_2d_array.zero_()
    
    # Launch single thread since we need sequential processing
    grid = (1,)
    s343_kernel[grid](
        aa, bb, flat_2d_array,
        len_2d
    )
    
    return flat_2d_array