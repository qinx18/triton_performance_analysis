import torch
import triton
import triton.language as tl

@triton.jit
def s119_kernel(aa_ptr, bb_ptr, diag_val, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # For diagonal diag_val, calculate valid range of i
    start_i = tl.maximum(1, diag_val - N + 1)
    end_i = tl.minimum(diag_val - 1, N - 1)
    
    # Calculate i and j for this block
    i = start_i + offsets
    j = diag_val - i
    
    # Create mask for valid elements
    mask = (i <= end_i) & (i >= 1) & (j >= 1) & (j < N)
    
    # Calculate indices
    aa_idx = i * N + j
    bb_idx = i * N + j
    prev_aa_idx = (i - 1) * N + (j - 1)
    
    # Load previous aa value and current bb value
    aa_prev = tl.load(aa_ptr + prev_aa_idx, mask=mask, other=0.0)
    bb_val = tl.load(bb_ptr + bb_idx, mask=mask, other=0.0)
    
    # Compute result and store
    result = aa_prev + bb_val
    tl.store(aa_ptr + aa_idx, result, mask=mask)

def s119_triton(aa, bb):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Process each anti-diagonal sequentially
    for diag in range(2, 2 * N):
        start_i = max(1, diag - N + 1)
        end_i = min(diag - 1, N - 1)
        
        if start_i <= end_i:
            num_elements = end_i - start_i + 1
            grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
            
            s119_kernel[grid](
                aa, bb, diag, N, BLOCK_SIZE=BLOCK_SIZE
            )