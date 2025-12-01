import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr):
    # This kernel processes the entire 2D arrays sequentially
    # due to the data dependency on k (output index)
    
    pid = tl.program_id(0)
    if pid > 0:
        return
    
    k = -1
    
    for i in range(LEN_2D):
        for j in range(LEN_2D):
            # Calculate linear index for 2D array access [j][i]
            idx = j * LEN_2D + i
            
            # Load bb[j][i]
            bb_val = tl.load(bb_ptr + idx)
            
            # Check condition
            if bb_val > 0.0:
                k += 1
                # Load aa[j][i] and store to flat_2d_array[k]
                aa_val = tl.load(aa_ptr + idx)
                tl.store(flat_2d_array_ptr + k, aa_val)

def s343_triton(aa, bb, flat_2d_array):
    LEN_2D = aa.shape[0]
    
    # Launch single thread since we have sequential dependency
    grid = (1,)
    
    s343_kernel[grid](
        aa, bb, flat_2d_array,
        LEN_2D=LEN_2D
    )
    
    return flat_2d_array