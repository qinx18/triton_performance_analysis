import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # This is a wavefront/anti-diagonal pattern that must be computed sequentially
    # For each anti-diagonal, we can parallelize within that diagonal
    
    for j in range(1, LEN_2D):
        # For each row j, compute all valid i values sequentially
        # since aa[j][i] depends on aa[j][i-1] in the same row
        for i in range(1, LEN_2D):
            # Calculate memory offset
            current_offset = j * LEN_2D + i
            left_offset = j * LEN_2D + (i - 1)  
            up_offset = (j - 1) * LEN_2D + i
            
            # Load values
            left_val = tl.load(aa_ptr + left_offset)
            up_val = tl.load(aa_ptr + up_offset)
            
            # Compute and store
            new_val = (left_val + up_val) / 1.9
            tl.store(aa_ptr + current_offset, new_val)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 32
    
    # Launch with single program since this must be sequential
    grid = (1,)
    s2111_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1
    )
    
    return aa