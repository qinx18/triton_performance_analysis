import torch
import triton
import triton.language as tl

@triton.jit
def s13110_kernel(aa_ptr, max_val_ptr, xindex_ptr, yindex_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < N
    
    # Initialize reduction variables
    local_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    local_xindex = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    local_yindex = j_idx
    
    # Sequential loop over i dimension
    for i in range(N):
        # Load current row values for valid j indices
        row_ptr = aa_ptr + i * N + j_idx
        values = tl.load(row_ptr, mask=j_mask, other=float('-inf'))
        
        # Update max values and indices where new values are greater
        is_greater = values > local_max
        local_max = tl.where(is_greater, values, local_max)
        local_xindex = tl.where(is_greater, i, local_xindex)
        local_yindex = tl.where(is_greater, j_idx, local_yindex)
    
    # Store results back to global memory
    tl.store(max_val_ptr + j_idx, local_max, mask=j_mask)
    tl.store(xindex_ptr + j_idx, local_xindex, mask=j_mask)
    tl.store(yindex_ptr + j_idx, local_yindex, mask=j_mask)

def s13110_triton(aa):
    N = aa.shape[0]
    BLOCK_SIZE = 256
    
    # Allocate temporary storage for partial results
    max_vals = torch.full((N,), float('-inf'), dtype=aa.dtype, device=aa.device)
    xindices = torch.zeros((N,), dtype=torch.int32, device=aa.device)
    yindices = torch.zeros((N,), dtype=torch.int32, device=aa.device)
    
    # Launch kernel
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    s13110_kernel[grid](
        aa, max_vals, xindices, yindices, N, BLOCK_SIZE
    )
    
    # Find global maximum from partial results
    global_max_idx = torch.argmax(max_vals)
    max_val = max_vals[global_max_idx].item()
    xindex = xindices[global_max_idx].item()
    yindex = yindices[global_max_idx].item()
    
    return max_val + xindex + 1 + yindex + 1