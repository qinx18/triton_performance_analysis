import torch
import triton
import triton.language as tl

@triton.jit
def s126_kernel(
    bb_ptr, cc_ptr, flat_2d_array_ptr,
    LEN_2D, 
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel for s126 computation.
    Each program handles one column (i) of the 2D array.
    """
    # Get program ID for column index
    i = tl.program_id(0)
    
    if i >= LEN_2D:
        return
    
    # Calculate starting k value for this column
    k_start = 1 + i * LEN_2D
    
    # Process rows sequentially for dependency handling
    for j in range(1, LEN_2D):
        # Calculate k index for this iteration
        k_idx = k_start + (j - 1)
        
        # Load values
        bb_prev = tl.load(bb_ptr + (j - 1) * LEN_2D + i)
        cc_val = tl.load(cc_ptr + j * LEN_2D + i)
        flat_val = tl.load(flat_2d_array_ptr + k_idx - 1)
        
        # Compute new value
        new_val = bb_prev + flat_val * cc_val
        
        # Store result
        tl.store(bb_ptr + j * LEN_2D + i, new_val)

def s126_triton(bb, cc, flat_2d_array):
    """
    Triton implementation of TSVC s126 function.
    Parallelizes across columns while maintaining row dependencies.
    """
    bb = bb.contiguous()
    cc = cc.contiguous() 
    flat_2d_array = flat_2d_array.contiguous()
    
    LEN_2D = bb.shape[0]
    
    # Launch kernel with one program per column
    # Block size doesn't matter much here since we process sequentially within each program
    BLOCK_SIZE = 64
    grid = (LEN_2D,)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return bb