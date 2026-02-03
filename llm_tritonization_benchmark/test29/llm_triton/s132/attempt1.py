import triton
import triton.language as tl

@triton.jit
def s132_kernel(aa_ptr, b_ptr, c_ptr, len_2d, j, k, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1  # Start from i=1
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < len_2d
    
    # Load c[1] once (scalar)
    c1_val = tl.load(c_ptr + 1)
    
    # Load b[i] values
    b_vals = tl.load(b_ptr + i_offsets, mask=mask)
    
    # Load aa[k][i-1] values
    prev_i_offsets = i_offsets - 1
    aa_k_prev_ptrs = aa_ptr + k * len_2d + prev_i_offsets
    aa_k_prev_vals = tl.load(aa_k_prev_ptrs, mask=mask)
    
    # Compute: aa[k][i-1] + b[i] * c[1]
    result = aa_k_prev_vals + b_vals * c1_val
    
    # Store to aa[j][i]
    aa_j_ptrs = aa_ptr + j * len_2d + i_offsets
    tl.store(aa_j_ptrs, result, mask=mask)

def s132_triton(aa, b, c, len_2d, j, k):
    BLOCK_SIZE = 256
    
    # Calculate grid size for range [1, len_2d)
    n_elements = len_2d - 1
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s132_kernel[grid](
        aa, b, c, len_2d, j, k,
        BLOCK_SIZE=BLOCK_SIZE
    )