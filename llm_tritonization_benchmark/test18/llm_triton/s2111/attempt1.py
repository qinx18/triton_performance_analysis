import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, N, stride_j, stride_i, BLOCK_SIZE: tl.constexpr):
    # Process elements sequentially since we have dependencies in both dimensions
    for j in range(1, N):
        # Process row j in blocks
        for i_start in range(1, N, BLOCK_SIZE):
            i_offsets = tl.arange(0, BLOCK_SIZE)
            i_indices = i_start + i_offsets
            mask = i_indices < N
            
            # Load aa[j][i-1] values
            i_minus_1 = i_indices - 1
            mask_i_minus_1 = (i_minus_1 >= 0) & (i_indices < N)
            aa_i_minus_1_ptrs = aa_ptr + j * stride_j + i_minus_1 * stride_i
            aa_i_minus_1 = tl.load(aa_i_minus_1_ptrs, mask=mask_i_minus_1, other=0.0)
            
            # Load aa[j-1][i] values
            aa_j_minus_1_ptrs = aa_ptr + (j - 1) * stride_j + i_indices * stride_i
            aa_j_minus_1 = tl.load(aa_j_minus_1_ptrs, mask=mask, other=0.0)
            
            # Compute new values
            new_vals = (aa_i_minus_1 + aa_j_minus_1) / 1.9
            
            # Store results
            aa_out_ptrs = aa_ptr + j * stride_j + i_indices * stride_i
            tl.store(aa_out_ptrs, new_vals, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    
    # Sequential processing due to dependencies in both dimensions
    BLOCK_SIZE = 32
    
    # Launch single kernel that processes everything sequentially
    grid = (1,)
    
    s2111_kernel[grid](
        aa,
        N,
        aa.stride(0),
        aa.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return aa