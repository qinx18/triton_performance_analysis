import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr):
    # Sequential computation due to data dependencies
    for j in range(1, LEN_2D):
        for i in range(1, LEN_2D):
            # Load aa[j][i-1]
            left_offset = j * LEN_2D + (i - 1)
            left_val = tl.load(aa_ptr + left_offset)
            
            # Load aa[j-1][i]
            top_offset = (j - 1) * LEN_2D + i
            top_val = tl.load(aa_ptr + top_offset)
            
            # Compute new value
            new_val = (left_val + top_val) / 1.9
            
            # Store aa[j][i]
            current_offset = j * LEN_2D + i
            tl.store(aa_ptr + current_offset, new_val)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    
    # Launch kernel with single program
    s2111_kernel[(1,)](
        aa, LEN_2D
    )
    
    return aa