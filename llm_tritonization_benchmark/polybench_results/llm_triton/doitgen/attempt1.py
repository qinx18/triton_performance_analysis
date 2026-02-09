import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A, C4, sum, NP, NQ, NR, BLOCK_P: tl.constexpr):
    # Get program IDs for parallelization over r and q dimensions
    pid_r = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    # Precompute offsets for vectorized operations
    p_offsets = tl.arange(0, BLOCK_P)
    s_offsets = tl.arange(0, BLOCK_P)
    
    # Process blocks of P dimension
    for p_start in range(0, NP, BLOCK_P):
        current_p_offsets = p_start + p_offsets
        p_mask = current_p_offsets < NP
        
        # Initialize sum values to 0
        sum_vals = tl.zeros([BLOCK_P], dtype=tl.float32)
        
        # Compute matrix multiplication: sum[p] = sum over s of A[r][q][s] * C4[s][p]
        for s_start in range(0, NP, BLOCK_P):
            current_s_offsets = s_start + s_offsets
            s_mask = current_s_offsets < NP
            
            # Load A[r][q][s] values
            a_indices = pid_r * (NQ * NP) + pid_q * NP + current_s_offsets
            a_vals = tl.load(A + a_indices, mask=s_mask, other=0.0)
            
            # For each p in current block, accumulate A[r][q][s] * C4[s][p]
            for p_idx in range(BLOCK_P):
                if p_start + p_idx < NP:
                    # Load C4[s][p] column
                    c4_indices = current_s_offsets * NP + (p_start + p_idx)
                    c4_vals = tl.load(C4 + c4_indices, mask=s_mask, other=0.0)
                    
                    # Compute dot product and accumulate
                    prod = a_vals * c4_vals
                    sum_vals = tl.where(p_offsets == p_idx, sum_vals + tl.sum(prod), sum_vals)
        
        # Store computed sum values
        sum_indices = current_p_offsets
        tl.store(sum + sum_indices, sum_vals, mask=p_mask)
        
        # Copy sum values back to A[r][q][p]
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