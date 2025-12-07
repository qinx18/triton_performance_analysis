import torch
import triton
import triton.language as tl

@triton.jit
def s3110_kernel(aa_ptr, max_vals_ptr, x_indices_ptr, y_indices_ptr, LEN_2D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_idx = pid * BLOCK_SIZE + j_offsets
    j_mask = j_idx < LEN_2D
    
    # Initialize with aa[0][0]
    init_val = tl.load(aa_ptr)
    max_val = init_val
    max_i = 0
    max_j = 0
    
    # Loop over all rows
    for i in range(LEN_2D):
        # Load values for current row and this block's j indices
        row_ptr = aa_ptr + i * LEN_2D + j_idx
        values = tl.load(row_ptr, mask=j_mask, other=init_val - 1.0)
        
        # Check each element in this block
        for k in range(BLOCK_SIZE):
            if j_idx[k] < LEN_2D:
                val = values[k]
                if val > max_val:
                    max_val = val
                    max_i = i
                    max_j = j_idx[k]
    
    # Store results from this block
    tl.store(max_vals_ptr + pid, max_val)
    tl.store(x_indices_ptr + pid, max_i)
    tl.store(y_indices_ptr + pid, max_j)

@triton.jit
def s3110_reduce_kernel(max_vals_ptr, x_indices_ptr, y_indices_ptr, 
                       final_max_ptr, final_x_ptr, final_y_ptr, 
                       num_blocks: tl.constexpr):
    max_val = tl.load(max_vals_ptr)
    max_i = tl.load(x_indices_ptr)
    max_j = tl.load(y_indices_ptr)
    
    for b in range(1, num_blocks):
        val = tl.load(max_vals_ptr + b)
        if val > max_val:
            max_val = val
            max_i = tl.load(x_indices_ptr + b)
            max_j = tl.load(y_indices_ptr + b)
    
    tl.store(final_max_ptr, max_val)
    tl.store(final_x_ptr, max_i)
    tl.store(final_y_ptr, max_j)

def s3110_triton(aa):
    LEN_2D = aa.shape[0]
    BLOCK_SIZE = 64
    num_blocks = triton.cdiv(LEN_2D, BLOCK_SIZE)
    
    # Temporary storage for partial results
    max_vals = torch.zeros(num_blocks, dtype=aa.dtype, device=aa.device)
    x_indices = torch.zeros(num_blocks, dtype=torch.int32, device=aa.device)
    y_indices = torch.zeros(num_blocks, dtype=torch.int32, device=aa.device)
    
    # Launch first kernel
    grid = (num_blocks,)
    s3110_kernel[grid](
        aa, max_vals, x_indices, y_indices,
        LEN_2D, BLOCK_SIZE
    )
    
    # Final result storage
    final_max = torch.zeros(1, dtype=aa.dtype, device=aa.device)
    final_x = torch.zeros(1, dtype=torch.int32, device=aa.device)
    final_y = torch.zeros(1, dtype=torch.int32, device=aa.device)
    
    # Reduce partial results
    grid = (1,)
    s3110_reduce_kernel[grid](
        max_vals, x_indices, y_indices,
        final_max, final_x, final_y,
        num_blocks
    )
    
    max_val = final_max.item()
    xindex = final_x.item()
    yindex = final_y.item()
    
    return max_val + xindex + 1 + yindex + 1