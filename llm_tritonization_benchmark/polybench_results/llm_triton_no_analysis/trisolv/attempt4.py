import torch
import triton
import triton.language as tl

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each block processes a chunk of rows
    row_start = pid * BLOCK_SIZE
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < N
    
    # Load b values for this block of rows
    b_vals = tl.load(b_ptr + row_offsets, mask=row_mask)
    
    # Initialize x values
    x_vals = b_vals
    
    # Process each column j < row_start (fully computed columns)
    for j in range(row_start):
        x_j = tl.load(x_ptr + j)
        L_offsets = row_offsets * N + j
        L_vals = tl.load(L_ptr + L_offsets, mask=row_mask)
        x_vals = x_vals - L_vals * x_j
    
    # Process rows within this block sequentially
    for local_i in range(BLOCK_SIZE):
        global_i = row_start + local_i
        if global_i >= N:
            break
            
        # Get current x value for this row
        x_i = tl.load(b_ptr + global_i)
        
        # Subtract contributions from all previous rows
        for j in range(global_i):
            L_ij = tl.load(L_ptr + global_i * N + j)
            x_j = tl.load(x_ptr + j)
            x_i = x_i - L_ij * x_j
        
        # Divide by diagonal element
        L_ii = tl.load(L_ptr + global_i * N + global_i)
        x_i = x_i / L_ii
        
        # Store result
        tl.store(x_ptr + global_i, x_i)

def trisolv_triton(L, b, x, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    trisolv_kernel[grid](L, b, x, N, BLOCK_SIZE)