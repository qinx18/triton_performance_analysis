import torch
import triton
import triton.language as tl

@triton.jit
def s232_kernel(aa_ptr, bb_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential execution due to data dependency
    for j in range(1, LEN_2D):
        # Process row j in blocks
        i_offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(1, j + 1, BLOCK_SIZE):
            block_end = min(block_start + BLOCK_SIZE, j + 1)
            current_i = block_start + i_offsets
            mask = (current_i >= block_start) & (current_i < block_end) & (current_i <= j)
            
            # Load aa[j][i-1]
            prev_ptrs = aa_ptr + j * LEN_2D + (current_i - 1)
            prev_vals = tl.load(prev_ptrs, mask=mask & (current_i > 1))
            
            # Load aa[j][i] for i=1 case (aa[j][0] doesn't get updated in loop)
            curr_ptrs = aa_ptr + j * LEN_2D + current_i
            curr_vals = tl.load(curr_ptrs, mask=mask & (current_i == 1))
            
            # Load bb[j][i]
            bb_ptrs = bb_ptr + j * LEN_2D + current_i
            bb_vals = tl.load(bb_ptrs, mask=mask)
            
            # Compute aa[j][i] = aa[j][i-1]*aa[j][i-1]+bb[j][i]
            # For i=1: use current aa[j][1], for i>1: use prev_vals
            aa_prev = tl.where(current_i == 1, curr_vals, prev_vals)
            result = aa_prev * aa_prev + bb_vals
            
            # Store result
            tl.store(curr_ptrs, result, mask=mask)

def s232_triton(aa, bb):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    
    grid = (1,)
    s232_kernel[grid](
        aa, bb,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )