import triton
import triton.language as tl
import torch

@triton.jit
def s172_kernel(a_ptr, b_ptr, n1, n3, N, BLOCK_SIZE: tl.constexpr):
    # Calculate starting position for this program
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset vector
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    valid_mask = indices < N
    
    # For each index, check if it matches the strided pattern
    # Original pattern: for (int i = n1-1; i < LEN_1D; i += n3)
    start_idx = n1 - 1
    
    # Check if indices are part of the strided sequence
    # An index i is valid if: i >= start_idx AND (i - start_idx) % n3 == 0
    stride_mask = (indices >= start_idx) & ((indices - start_idx) % n3 == 0)
    
    # Combine both masks
    final_mask = valid_mask & stride_mask
    
    # Load values where mask is true
    a_vals = tl.load(a_ptr + indices, mask=final_mask, other=0.0)
    b_vals = tl.load(b_ptr + indices, mask=final_mask, other=0.0)
    
    # Compute result
    result = a_vals + b_vals
    
    # Store result back
    tl.store(a_ptr + indices, result, mask=final_mask)

def s172_triton(a, b, n1, n3):
    N = a.shape[0]
    BLOCK_SIZE = 256
    
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    grid = (grid_size,)
    
    s172_kernel[grid](
        a, b, n1, n3, N,
        BLOCK_SIZE=BLOCK_SIZE
    )