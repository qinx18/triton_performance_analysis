import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A, C4, sum, NP, NQ, NR, BLOCK_P: tl.constexpr):
    # Get program IDs for parallelization over r and q dimensions
    pid_r = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    # Define offsets once
    p_offsets = tl.arange(0, BLOCK_P)
    
    # Process blocks of P dimension
    for p_start in range(0, NP, BLOCK_P):
        current_p_offsets = p_start + p_offsets
        p_mask = current_p_offsets < NP
        
        # Initialize sum values to 0
        sum_vals = tl.zeros([BLOCK_P], dtype=tl.float32)
        
        # Compute matrix multiplication: sum[p] = sum over s of A[r][q][s] * C4[s][p]
        for s in range(NP):
            # Load A[r][q][s] (scalar value)
            a_idx = pid_r * (NQ * NP) + pid_q * NP + s
            a_val = tl.load(A + a_idx)
            
            # Load C4[s][p] values for current block
            c4_indices = s * NP + current_p_offsets
            c4_vals = tl.load(C4 + c4_indices, mask=p_mask, other=0.0)
            
            # Accumulate A[r][q][s] * C4[s][p]
            sum_vals += a_val * c4_vals
        
        # Store computed sum values to global sum array
        sum_indices = current_p_offsets
        tl.store(sum + sum_indices, sum_vals, mask=p_mask)
        
        # Copy sum values to A[r][q][p]
        a_out_indices = pid_r * (NQ * NP) + pid_q * NP + current_p_offsets
        tl.store(A + a_out_indices, sum_vals, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    BLOCK_P = 32
    
    # Launch kernel with grid over (r, q) dimensions
    grid = (NR, NQ)
    
    doitgen_kernel[grid](
        A, C4, sum,
        NP, NQ, NR,
        BLOCK_P=BLOCK_P
    )