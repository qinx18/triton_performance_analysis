import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the i dimension
    pid_i = tl.program_id(0)
    
    # Check bounds for i
    if pid_i >= len_2d:
        return
    
    # For each i, we need to process j from 1 to len_2d-1 sequentially
    # due to the recurrence bb[j][i] = bb[j-1][i] + ...
    
    k = pid_i * len_2d + 1
    
    for j in range(1, len_2d):
        # Load bb[j-1][i]
        bb_prev_offset = (j - 1) * len_2d + pid_i
        bb_prev = tl.load(bb_ptr + bb_prev_offset)
        
        # Load flat_2d_array[k-1]
        flat_val = tl.load(flat_2d_array_ptr + k - 1)
        
        # Load cc[j][i]
        cc_offset = j * len_2d + pid_i
        cc_val = tl.load(cc_ptr + cc_offset)
        
        # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
        result = bb_prev + flat_val * cc_val
        
        # Store bb[j][i]
        bb_offset = j * len_2d + pid_i
        tl.store(bb_ptr + bb_offset, result)
        
        k += 1
    
    # Increment k one more time as in the original loop
    k += 1

def s126_triton(bb, cc, flat_2d_array, len_2d):
    # Ensure tensors are contiguous
    bb = bb.contiguous()
    cc = cc.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    
    # Launch kernel with one thread per i dimension
    BLOCK_SIZE = 256
    grid = (len_2d,)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return bb