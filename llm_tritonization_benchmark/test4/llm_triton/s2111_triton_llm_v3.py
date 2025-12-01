import triton
import triton.language as tl
import torch

@triton.jit
def s2111_kernel(aa_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Sequential wavefront computation - cannot be parallelized due to dependencies
    for j in range(1, LEN_2D):
        block_start = 0
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(1, LEN_2D, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < LEN_2D
            
            # Load aa[j][i-1]
            left_ptrs = aa_ptr + j * LEN_2D + (current_offsets - 1)
            left_vals = tl.load(left_ptrs, mask=mask)
            
            # Load aa[j-1][i]
            top_ptrs = aa_ptr + (j - 1) * LEN_2D + current_offsets
            top_vals = tl.load(top_ptrs, mask=mask)
            
            # Compute new values
            new_vals = (left_vals + top_vals) / 1.9
            
            # Store aa[j][i]
            out_ptrs = aa_ptr + j * LEN_2D + current_offsets
            tl.store(out_ptrs, new_vals, mask=mask)

def s2111_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = min(256, triton.next_power_of_2(LEN_2D))
    
    grid = (1,)
    s2111_kernel[grid](
        aa,
        LEN_2D=LEN_2D,
        BLOCK_SIZE=BLOCK_SIZE
    )