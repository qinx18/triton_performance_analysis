import triton
import triton.language as tl
import torch

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < (N * N)
    
    # Convert linear indices to 2D indices
    i_indices = offsets // N
    j_indices = offsets % N
    
    # Calculate 2D array offsets
    aa_offsets = i_indices * N + j_indices
    bb_offsets = i_indices * N + j_indices
    cc_offsets = i_indices * N + j_indices
    
    # Load values
    aa_vals = tl.load(aa_ptr + aa_offsets, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask)
    cc_vals = tl.load(cc_ptr + cc_offsets, mask=mask)
    
    # Compute result
    result = aa_vals + bb_vals * cc_vals
    
    # Store to flat array
    tl.store(flat_2d_array_ptr + offsets, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    N = aa.shape[0]
    total_elements = N * N
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array,
        N, BLOCK_SIZE=BLOCK_SIZE
    )