import triton
import triton.language as tl
import torch

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr):
    # This kernel handles the triangular loop pattern sequentially
    # due to the dependency aa[j][i] = aa[j][i-1]*aa[j][i-1]+bb[j][i]
    
    for j in range(1, LEN_2D):
        for i in range(1, j + 1):
            # Calculate linear indices for 2D arrays
            current_idx = j * LEN_2D + i
            prev_idx = j * LEN_2D + (i - 1)
            
            # Load previous value and current bb value
            aa_prev = tl.load(aa_ptr + prev_idx)
            bb_val = tl.load(bb_ptr + current_idx)
            
            # Compute and store result
            result = aa_prev * aa_prev + bb_val
            tl.store(aa_ptr + current_idx, result)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with single program
    grid = (1,)
    s232_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D
    )
    
    return aa