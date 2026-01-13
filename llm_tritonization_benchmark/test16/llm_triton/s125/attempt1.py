import triton
import triton.language as tl
import torch

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, N, BLOCK_SIZE: tl.constexpr):
    total_elements = N * N
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, total_elements, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < total_elements
        
        # Load elements
        aa_vals = tl.load(aa_ptr + current_offsets, mask=mask)
        bb_vals = tl.load(bb_ptr + current_offsets, mask=mask)
        cc_vals = tl.load(cc_ptr + current_offsets, mask=mask)
        
        # Compute: aa[k] + bb[k] * cc[k]
        result = aa_vals + bb_vals * cc_vals
        
        # Store result
        tl.store(flat_2d_array_ptr + current_offsets, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    N = aa.shape[0]
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid = (1,)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )