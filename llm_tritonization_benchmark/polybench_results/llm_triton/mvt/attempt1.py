import triton
import triton.language as tl
import torch

@triton.jit
def mvt_kernel(
    A_ptr, x1_ptr, x2_ptr, y_1_ptr, y_2_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # First loop: x1[i] = x1[i] + A[i][j] * y_1[j]
    if pid < N:
        i = pid
        
        # Load initial x1[i] value
        x1_val = tl.load(x1_ptr + i)
        
        # Vectorized reduction over j dimension
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, N, BLOCK_SIZE):
            j_idx = j_start + j_offsets
            j_mask = j_idx < N
            
            # Load A[i][j] values
            A_offsets = i * N + j_idx
            A_vals = tl.load(A_ptr + A_offsets, mask=j_mask, other=0.0)
            
            # Load y_1[j] values
            y_1_vals = tl.load(y_1_ptr + j_idx, mask=j_mask, other=0.0)
            
            # Accumulate A[i][j] * y_1[j]
            products = A_vals * y_1_vals
            x1_val += tl.sum(products)
        
        # Store updated x1[i]
        tl.store(x1_ptr + i, x1_val)
    
    # Second loop: x2[i] = x2[i] + A[j][i] * y_2[j]
    if pid < N:
        i = pid
        
        # Load initial x2[i] value
        x2_val = tl.load(x2_ptr + i)
        
        # Vectorized reduction over j dimension
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, N, BLOCK_SIZE):
            j_idx = j_start + j_offsets
            j_mask = j_idx < N
            
            # Load A[j][i] values (transposed access)
            A_offsets = j_idx * N + i
            A_vals = tl.load(A_ptr + A_offsets, mask=j_mask, other=0.0)
            
            # Load y_2[j] values
            y_2_vals = tl.load(y_2_ptr + j_idx, mask=j_mask, other=0.0)
            
            # Accumulate A[j][i] * y_2[j]
            products = A_vals * y_2_vals
            x2_val += tl.sum(products)
        
        # Store updated x2[i]
        tl.store(x2_ptr + i, x2_val)

def mvt_triton(A, x1, x2, y_1, y_2, N):
    BLOCK_SIZE = 64
    grid = (N,)
    
    mvt_kernel[grid](
        A, x1, x2, y_1, y_2,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )