import torch
import triton
import triton.language as tl

@triton.jit
def s141_kernel(bb_ptr, flat_2d_array_ptr, LEN_2D, BLOCK_SIZE_I: tl.constexpr):
    # Get program ID for i dimension
    pid_i = tl.program_id(0)
    i_start = pid_i * BLOCK_SIZE_I
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    i_indices = i_start + i_offsets
    i_mask = i_indices < LEN_2D
    
    # Process each i in this block
    for i_idx in range(BLOCK_SIZE_I):
        i = i_start + i_idx
        valid = i < LEN_2D
        
        if valid:
            # Calculate initial k for this i
            k = (i + 1) * i // 2 + i
            
            # Inner loop over j from i to LEN_2D-1
            for j in range(i, LEN_2D):
                # Load bb[j][i]
                bb_val = tl.load(bb_ptr + j * LEN_2D + i)
                
                # Load current flat_2d_array[k]
                current_val = tl.load(flat_2d_array_ptr + k)
                
                # Update flat_2d_array[k]
                new_val = current_val + bb_val
                tl.store(flat_2d_array_ptr + k, new_val)
                
                # Update k for next iteration
                k += j + 1

def s141_triton(bb, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    # Launch kernel with appropriate grid
    BLOCK_SIZE_I = 32
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE_I),)
    
    s141_kernel[grid](
        bb, flat_2d_array,
        LEN_2D,
        BLOCK_SIZE_I
    )