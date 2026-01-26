import triton
import triton.language as tl
import torch

@triton.jit
def s126_kernel(bb_ptr, cc_ptr, flat_2d_array_ptr, len_2d, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= len_2d:
        return
    
    # Process column i sequentially
    k = 1 + i * len_2d  # Starting k for this column
    
    # First iteration (j=1) - bb[0][i] is the previous value
    if len_2d > 1:
        prev_val = tl.load(bb_ptr + 0 * len_2d + i)
        flat_val = tl.load(flat_2d_array_ptr + k - 1)
        cc_val = tl.load(cc_ptr + 1 * len_2d + i)
        new_val = prev_val + flat_val * cc_val
        tl.store(bb_ptr + 1 * len_2d + i, new_val)
        k += 1
        
        # Remaining iterations (j=2 to len_2d-1)
        for j in range(2, len_2d):
            prev_val = tl.load(bb_ptr + (j - 1) * len_2d + i)
            flat_val = tl.load(flat_2d_array_ptr + k - 1)
            cc_val = tl.load(cc_ptr + j * len_2d + i)
            new_val = prev_val + flat_val * cc_val
            tl.store(bb_ptr + j * len_2d + i, new_val)
            k += 1

def s126_triton(bb, cc, flat_2d_array, len_2d):
    BLOCK_SIZE = 256
    grid = (triton.cdiv(len_2d, 1),)
    
    s126_kernel[grid](
        bb, cc, flat_2d_array, len_2d,
        BLOCK_SIZE=BLOCK_SIZE
    )