import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D, BLOCK_SIZE_I: tl.constexpr):
    pid_i = tl.program_id(0)
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    i_start = pid_i * BLOCK_SIZE_I
    i_indices = i_start + i_offsets
    i_mask = i_indices < LEN_2D
    
    # Process each valid i in this block
    for i_offset in range(BLOCK_SIZE_I):
        i = i_start + i_offset
        
        # Check if i is within bounds
        i_valid = i < LEN_2D
        
        if i_valid:
            k = 1 + i * (LEN_2D - 1) + i
            
            # Sequential loop over j (due to bb[j] = bb[j-1] dependency)
            for j in range(1, LEN_2D):
                # Load bb[j-1][i]
                bb_prev_ptr = bb_ptr + (j-1) * LEN_2D + i
                bb_prev = tl.load(bb_prev_ptr)
                
                # Load flat_2d_array[k-1]
                flat_ptr = flat_2d_array_ptr + (k-1)
                flat_val = tl.load(flat_ptr)
                
                # Load cc[j][i]
                cc_ptr_idx = cc_ptr + j * LEN_2D + i
                cc_val = tl.load(cc_ptr_idx)
                
                # Compute bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
                result = bb_prev + flat_val * cc_val
                
                # Store bb[j][i]
                bb_ptr_idx = bb_ptr + j * LEN_2D + i
                tl.store(bb_ptr_idx, result)
                
                k += 1

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE_I = min(32, LEN_2D)
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE_I),)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D,
        BLOCK_SIZE_I
    )