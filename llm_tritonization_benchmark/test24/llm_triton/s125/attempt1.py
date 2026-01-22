import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, N, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N * N
    
    # Convert linear index to 2D indices
    i = offsets // N
    j = offsets % N
    
    # Calculate 2D array indices
    aa_indices = i * N + j
    bb_indices = i * N + j
    cc_indices = i * N + j
    
    # Load values from 2D arrays
    aa_vals = tl.load(aa_ptr + aa_indices, mask=mask)
    bb_vals = tl.load(bb_ptr + bb_indices, mask=mask)
    cc_vals = tl.load(cc_ptr + cc_indices, mask=mask)
    
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