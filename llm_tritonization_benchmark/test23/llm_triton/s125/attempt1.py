import triton
import triton.language as tl

@triton.jit
def s125_kernel(aa_ptr, bb_ptr, cc_ptr, flat_2d_array_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    k_offsets = block_start + offsets
    mask = k_offsets < N * N
    
    # Convert linear index k to 2D indices (i, j)
    i_indices = k_offsets // N
    j_indices = k_offsets % N
    
    # Calculate 2D array offsets
    array_2d_offsets = i_indices * N + j_indices
    
    # Load data
    aa_vals = tl.load(aa_ptr + array_2d_offsets, mask=mask, other=0.0)
    bb_vals = tl.load(bb_ptr + array_2d_offsets, mask=mask, other=0.0)
    cc_vals = tl.load(cc_ptr + array_2d_offsets, mask=mask, other=0.0)
    
    # Compute: aa[i][j] + bb[i][j] * cc[i][j]
    result = aa_vals + bb_vals * cc_vals
    
    # Store result
    tl.store(flat_2d_array_ptr + k_offsets, result, mask=mask)

def s125_triton(aa, bb, cc, flat_2d_array):
    N = aa.shape[0]
    total_elements = N * N
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    s125_kernel[grid](
        aa, bb, cc, flat_2d_array,
        N, BLOCK_SIZE
    )