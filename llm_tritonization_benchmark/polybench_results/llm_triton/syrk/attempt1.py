import torch
import triton
import triton.language as tl

@triton.jit
def syrk_kernel(
    C_ptr, A_ptr,
    alpha, beta,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)
    
    # First phase: scale existing C row by beta
    col_offs = tl.arange(0, BLOCK_N)
    
    for j_start in range(0, row + 1, BLOCK_N):
        current_col_offs = j_start + col_offs
        mask = (current_col_offs < N) & (current_col_offs <= row)
        
        c_vals = tl.load(C_ptr + row * N + current_col_offs, mask=mask)
        c_vals = beta * c_vals
        
        # Second phase: accumulate A[row, k] * A[j, k] for all k
        for k in range(M):
            a_row_k = tl.load(A_ptr + row * M + k)
            a_col_k = tl.load(A_ptr + current_col_offs * M + k, mask=mask)
            c_vals += alpha * a_row_k * a_col_k
        
        tl.store(C_ptr + row * N + current_col_offs, c_vals, mask=mask)

def syrk_triton(A, C, alpha, beta, M, N):
    BLOCK_N = min(triton.next_power_of_2(N), 128)
    
    grid = (N,)
    
    syrk_kernel[grid](
        C, A,
        alpha, beta,
        M=M, N=N,
        BLOCK_N=BLOCK_N
    )