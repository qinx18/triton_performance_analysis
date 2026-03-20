import triton
import triton.language as tl
import torch

@triton.jit
def mvt_kernel(A_ptr, x1_ptr, x2_ptr, y_1_ptr, y_2_ptr, 
               N: tl.constexpr,
               BLOCK_SIZE: tl.constexpr):
    
    # Get program ID for the current block
    pid = tl.program_id(axis=0)
    
    # Calculate the row index for this block
    row_start = pid * BLOCK_SIZE
    row_offsets = tl.arange(0, BLOCK_SIZE)
    row_indices = row_start + row_offsets
    row_mask = row_indices < N
    
    # Initialize accumulators for x1 and x2
    acc_x1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    acc_x2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Load initial values of x1 and x2
    x1_vals = tl.load(x1_ptr + row_indices, mask=row_mask, other=0.0)
    x2_vals = tl.load(x2_ptr + row_indices, mask=row_mask, other=0.0)
    
    # Column offsets for vectorized access
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process columns in blocks
    for col_start in range(0, N, BLOCK_SIZE):
        col_indices = col_start + col_offsets
        col_mask = col_indices < N
        
        # Load y_1 and y_2 values for current column block
        y_1_vals = tl.load(y_1_ptr + col_indices, mask=col_mask, other=0.0)
        y_2_vals = tl.load(y_2_ptr + col_indices, mask=col_mask, other=0.0)
        
        # For each row in the current row block
        for r in range(BLOCK_SIZE):
            if row_start + r >= N:
                break
                
            # Load A[row, :] for first computation (x1)
            a_row_indices = (row_start + r) * N + col_indices
            a_vals = tl.load(A_ptr + a_row_indices, mask=col_mask, other=0.0)
            
            # Compute contribution to x1[row]
            products = a_vals * y_1_vals
            valid_mask = col_mask
            acc_x1 = tl.where(tl.arange(0, BLOCK_SIZE) == r, 
                             acc_x1 + tl.sum(tl.where(valid_mask, products, 0.0)), 
                             acc_x1)
            
            # Load A[:, row] for second computation (x2) - transpose access
            a_col_indices = col_indices * N + (row_start + r)
            a_t_vals = tl.load(A_ptr + a_col_indices, mask=col_mask, other=0.0)
            
            # Compute contribution to x2[row]
            products_t = a_t_vals * y_2_vals
            acc_x2 = tl.where(tl.arange(0, BLOCK_SIZE) == r,
                             acc_x2 + tl.sum(tl.where(valid_mask, products_t, 0.0)),
                             acc_x2)
    
    # Add accumulated values to original x1 and x2
    final_x1 = x1_vals + acc_x1
    final_x2 = x2_vals + acc_x2
    
    # Store results
    tl.store(x1_ptr + row_indices, final_x1, mask=row_mask)
    tl.store(x2_ptr + row_indices, final_x2, mask=row_mask)

def mvt_triton(A, x1, x2, y_1, y_2, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    mvt_kernel[grid](
        A, x1, x2, y_1, y_2,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )