import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(
    aa_ptr,
    bb_ptr,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s232 - processes triangular dependency pattern.
    Each program handles one row j, processing elements sequentially within that row.
    """
    j = tl.program_id(0) + 1  # j starts from 1
    
    if j >= LEN_2D:
        return
    
    # Process elements in row j from i=1 to i=j sequentially
    # This preserves the dependency: aa[j,i] depends on aa[j,i-1]
    for i in range(1, j + 1):
        aa_offset = j * LEN_2D + i
        bb_offset = j * LEN_2D + i
        prev_aa_offset = j * LEN_2D + (i - 1)
        
        # Load previous aa value and current bb value
        prev_aa_val = tl.load(aa_ptr + prev_aa_offset)
        bb_val = tl.load(bb_ptr + bb_offset)
        
        # Compute: aa[j,i] = aa[j,i-1]^2 + bb[j,i]
        new_aa_val = prev_aa_val * prev_aa_val + bb_val
        
        # Store result
        tl.store(aa_ptr + aa_offset, new_aa_val)

def s232_triton(aa, bb):
    """
    Triton implementation of TSVC s232.
    
    Optimizations:
    - Each thread block processes one row to maintain sequential dependency
    - Memory coalescing within rows where possible
    - Reduced kernel launch overhead by processing multiple elements per thread
    """
    aa = aa.contiguous()
    bb = bb.contiguous()
    
    LEN_2D = aa.shape[0]
    
    # Launch one program per row (j from 1 to LEN_2D-1)
    grid = (LEN_2D - 1,)
    BLOCK_SIZE = 256
    
    s232_kernel[grid](
        aa,
        bb,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return aa