import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(
    bb_ptr,
    flat_2d_array_ptr,
    LEN_2D,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s141 computation.
    Each program handles one value of i across all j values.
    """
    # Get program ID for the i dimension
    pid = tl.program_id(0)
    
    if pid >= LEN_2D:
        return
    
    i = pid
    
    # Calculate initial k value: k = (i+1) * i / 2 + i
    k = (i + 1) * i // 2 + i
    
    # Process all j values from i to LEN_2D-1
    for j in range(i, LEN_2D):
        # Load bb[j, i]
        bb_idx = j * LEN_2D + i
        bb_val = tl.load(bb_ptr + bb_idx)
        
        # Load current flat_2d_array[k] value
        current_val = tl.load(flat_2d_array_ptr + k)
        
        # Update flat_2d_array[k]
        new_val = current_val + bb_val
        tl.store(flat_2d_array_ptr + k, new_val)
        
        # Update k for next iteration: k += j + 1
        k += j + 1

def s141_triton(bb, flat_2d_array):
    """
    Triton implementation of TSVC s141
    
    Optimizations:
    - Parallelizes across the outer i loop
    - Sequential processing of j loop to maintain dependency ordering
    - Direct memory access patterns for optimal throughput
    """
    bb = bb.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    LEN_2D = bb.shape[0]
    
    # Launch configuration - one thread per i value
    BLOCK_SIZE = 1
    grid = (LEN_2D,)
    
    # Launch kernel
    s141_kernel[grid](
        bb,
        flat_2d_array,
        LEN_2D,
        BLOCK_SIZE,
    )
    
    return bb, flat_2d_array