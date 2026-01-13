import torch
import triton
import triton.language as tl

@triton.jit
def s2111_kernel(aa_ptr, N, stride_j, stride_i, diag, start_j, end_j, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = start_j + block_start + offsets
    mask = (j_indices < end_j) & (j_indices >= start_j)
    
    # For diagonal processing: j + i = diag, so i = diag - j
    i_indices = diag - j_indices
    
    # Load aa[j][i-1]
    i_minus_1 = i_indices - 1
    aa_i_minus_1_ptrs = aa_ptr + j_indices * stride_j + i_minus_1 * stride_i
    aa_i_minus_1 = tl.load(aa_i_minus_1_ptrs, mask=mask, other=0.0)
    
    # Load aa[j-1][i]
    j_minus_1 = j_indices - 1
    aa_j_minus_1_ptrs = aa_ptr + j_minus_1 * stride_j + i_indices * stride_i
    aa_j_minus_1 = tl.load(aa_j_minus_1_ptrs, mask=mask, other=0.0)
    
    # Compute new values
    new_vals = (aa_i_minus_1 + aa_j_minus_1) / 1.9
    
    # Store results
    aa_out_ptrs = aa_ptr + j_indices * stride_j + i_indices * stride_i
    tl.store(aa_out_ptrs, new_vals, mask=mask)

def s2111_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 128
    
    # Process anti-diagonals sequentially
    for diag in range(2, 2 * N - 1):
        start_j = max(1, diag - N + 1)
        end_j = min(diag, N)
        
        num_elements = end_j - start_j
        if num_elements <= 0:
            continue
            
        num_blocks = triton.cdiv(num_elements, BLOCK_SIZE)
        grid = (num_blocks,)
        
        s2111_kernel[grid](
            aa,
            N,
            aa.stride(0),
            aa.stride(1),
            diag,
            start_j,
            end_j,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    return aa