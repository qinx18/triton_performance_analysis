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
            # Calculate linear index for 2D arrays [j][i]
            idx_2d = j * len_2d + i
            
            # Load bb[j][i]
            bb_val = tl.load(bb_ptr + idx_2d)
            
            # Check condition
            if bb_val > 0.0:
                k += 1
                # Load aa[j][i] and store to flat_2d_array[k]
                aa_val = tl.load(aa_ptr + idx_2d)
                tl.store(flat_2d_array_ptr + k, aa_val)

def s343_triton(aa, bb, flat_2d_array):
    len_2d = aa.shape[0]
    
    # Launch single thread since this is inherently sequential
    grid = (1,)
    
    s343_kernel[grid](
        aa, bb, flat_2d_array, len_2d
    )
    
    return flat_2d_array