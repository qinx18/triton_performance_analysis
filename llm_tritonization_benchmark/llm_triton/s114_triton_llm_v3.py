import triton
import triton.language as tl
import torch

@triton.jit
def s114_kernel(aa_ptr, bb_ptr, LEN_2D, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Process elements sequentially since parallelization causes race conditions
    for i in range(LEN_2D):
        for j in range(i):
            if pid == 0:  # Only first thread does the work
                # Read values
                aa_ji = tl.load(aa_ptr + j * LEN_2D + i)
                bb_ij = tl.load(bb_ptr + i * LEN_2D + j)
                
                # Compute and store
                result = aa_ji + bb_ij
                tl.store(aa_ptr + i * LEN_2D + j, result)

def s114_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Launch with single program to avoid race conditions
    grid = (1,)
    
    s114_kernel[grid](
        aa, bb, LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa