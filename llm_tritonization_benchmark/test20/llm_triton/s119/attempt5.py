import triton
import triton.language as tl
import torch

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag, start_i, end_i, N, stride_aa_0, stride_aa_1, stride_bb_0, stride_bb_1, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = start_i + block_start + offsets
    
    mask = (i_indices < end_i) & (i_indices >= start_i) & (i_indices < N)
    j_indices = diag - i_indices
    
    # Check bounds for j (j should be >= 1 and < N)
    valid_j = (j_indices >= 1) & (j_indices < N)
    mask = mask & valid_j
    
    # Load aa[i-1][j-1]
    aa_prev_offsets = (i_indices - 1) * stride_aa_0 + (j_indices - 1) * stride_aa_1
    aa_prev_vals = tl.load(aa_ptr + aa_prev_offsets, mask=mask, other=0.0)
    
    # Load bb[i][j]
    bb_offsets = i_indices * stride_bb_0 + j_indices * stride_bb_1
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)
    
    # Compute result
    result = aa_prev_vals + bb_vals
    
    # Store aa[i][j]
    aa_offsets = i_indices * stride_aa_0 + j_indices * stride_aa_1
    tl.store(aa_ptr + aa_offsets, result, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    
    stride_aa_0 = aa.stride(0)
    stride_aa_1 = aa.stride(1)
    stride_bb_0 = bb.stride(0)
    stride_bb_1 = bb.stride(1)
    
    BLOCK_SIZE = 256
    
    # Process wavefront/anti-diagonal pattern
    for diag in range(2, 2 * N):
        start_i = max(1, diag - N + 1)
        end_i = min(diag, N)
        
        if start_i >= end_i:
            continue
            
        num_elements = end_i - start_i
        grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
        
        s119_kernel[grid](
            aa, bb, diag, start_i, end_i, N,
            stride_aa_0, stride_aa_1, stride_bb_0, stride_bb_1, BLOCK_SIZE
        )