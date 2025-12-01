import torch
import triton
import triton.language as tl

@triton.jit
def s343_kernel(aa_ptr, bb_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # This kernel handles the packing operation sequentially
    # Each program handles one complete iteration of the nested loops
    
    pid = tl.program_id(0)
    if pid > 0:  # Only use one program since we need sequential packing
        return
    
    # Initialize k counter
    k = -1
    
    # Sequential iteration over the 2D arrays
    for i in range(len_2d):
        for j in range(len_2d):
            # Calculate linear index for 2D arrays (j, i)
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
    
    # Reset the flat array
    flat_2d_array.zero_()
    
    # Launch kernel with single thread block since packing is inherently sequential
    grid = (1,)
    BLOCK_SIZE = 256
    
    s343_kernel[grid](
        aa, bb, flat_2d_array,
        len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return flat_2d_array