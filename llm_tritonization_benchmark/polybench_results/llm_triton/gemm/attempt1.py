import triton
import triton.language as tl

@triton.jit
def gemm_kernel(C_ptr, A_ptr, B_ptr, NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, 
                alpha, beta, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    
    # Load column offsets once
    col_offs = tl.arange(0, BLOCK_N)
    mask = col_offs < NJ
    
    # Scale existing C row by beta
    c_row = tl.load(C_ptr + row * NJ + col_offs, mask=mask)
    c_row = beta * c_row
    
    # Accumulate A[row,:] @ B into c_row
    for k in range(NK):
        a_val = tl.load(A_ptr + row * NK + k)
        b_row = tl.load(B_ptr + k * NJ + col_offs, mask=mask)
        c_row += alpha * a_val * b_row
    
    # Store result
    tl.store(C_ptr + row * NJ + col_offs, c_row, mask=mask)

def gemm_triton(A, B, C, alpha, beta, NI, NJ, NK):
    BLOCK_N = min(triton.next_power_of_2(NJ), 128)
    grid = (NI,)
    
    gemm_kernel[grid](
        C, A, B,
        NI, NJ, NK,
        alpha, beta,
        BLOCK_N
    )