import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the outer i loop
    pid = tl.program_id(axis=0)
    i = pid
    
    if i >= len_2d:
        return
    
    # Sequential j loop from 1 to len_2d-1
    k = i * len_2d + 1
    
    for j in range(1, len_2d):
        # Load bb[j-1][i]
        bb_prev_offset = (j - 1) * len_2d + i
        bb_prev = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1]
        flat_val = tl.load(flat_2d_array_ptr + (k - 1))
        
        # Load cc[j][i]
        cc_offset = j * len_2d + i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = bb_prev + flat_val * cc_val
        
        # Store bb[j][i]
        bb_offset = j * len_2d + i
        tl.store(bb_ptr + bb_offset, result)
        
        k += 1
    
def s126_triton(bb, cc, flat_2d_array, len_2d):
    BLOCK_SIZE = 256
    
    # Launch kernel with one thread per i value
    grid = (triton.cdiv(len_2d, 1),)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )