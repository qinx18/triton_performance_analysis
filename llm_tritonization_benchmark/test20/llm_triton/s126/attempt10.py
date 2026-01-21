import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE_I: tl.constexpr):
    pid_i = tl.program_id(0)
    
    # Each program handles BLOCK_SIZE_I columns
    i_start = pid_i * BLOCK_SIZE_I
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    i_mask = i_offsets < LEN_2D
    
    # Process each valid column i
    for i_idx in range(BLOCK_SIZE_I):
        i = i_start + i_idx
        i_valid = i < LEN_2D
        
        # Skip if i is out of bounds, but don't return (just continue)
        if i_valid:
            k = 1 + i * LEN_2D  # k starts at 1 and gets incremented by LEN_2D per outer loop iteration
            
            for j in range(1, LEN_2D):
                # bb[j][i] = bb[j-1][i] + flat_2d_array[k-1] * cc[j][i]
                bb_prev_offset = (j - 1) * LEN_2D + i
                bb_prev = tl.load(bb_ptr + bb_prev_offset)
                
                flat_offset = k - 1
                flat_val = tl.load(flat_2d_array_ptr + flat_offset)
                
                cc_offset = j * LEN_2D + i
                cc_val = tl.load(cc_ptr + cc_offset)
                
                result = bb_prev + flat_val * cc_val
                
                bb_offset = j * LEN_2D + i
                tl.store(bb_ptr + bb_offset, result)
                
                k += 1
            
            # After inner j loop, increment k one more time (as in original C code)
            k += 1

def s126_triton(bb, cc, flat_2d_array):
    LEN_2D = bb.shape[0]
    
    BLOCK_SIZE_I = 64
    grid = (triton.cdiv(LEN_2D, BLOCK_SIZE_I),)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array,
        LEN_2D=LEN_2D,
        BLOCK_SIZE_I=BLOCK_SIZE_I
    )