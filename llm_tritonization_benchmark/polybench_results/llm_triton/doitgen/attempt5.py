import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A, C4, sum, NP, NQ, NR, BLOCK_R: tl.constexpr, BLOCK_Q: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    r_start = pid_r * BLOCK_R
    q_start = pid_q * BLOCK_Q
    
    for r_idx in tl.static_range(BLOCK_R):
        r = r_start + r_idx
        r_is_valid = r < NR
        
        for q_idx in tl.static_range(BLOCK_Q):
            q = q_start + q_idx
            q_is_valid = q < NQ
            
            valid = r_is_valid & q_is_valid
            
            if valid:
                # Compute sum[p] for each p
                for p in range(NP):
                    sum_val = 0.0
                    for s in range(NP):
                        a_idx = r * (NQ * NP) + q * NP + s
                        a_val = tl.load(A + a_idx)
                        
                        c4_idx = s * NP + p
                        c4_val = tl.load(C4 + c4_idx)
                        
                        sum_val += a_val * c4_val
                    
                    # Store sum_val back to A[r][q][p]
                    a_out_idx = r * (NQ * NP) + q * NP + p
                    tl.store(A + a_out_idx, sum_val)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    BLOCK_R = 4
    BLOCK_Q = 4
    
    grid_r = triton.cdiv(NR, BLOCK_R)
    grid_q = triton.cdiv(NQ, BLOCK_Q)
    
    doitgen_kernel[(grid_r, grid_q)](
        A, C4, sum, NP, NQ, NR, BLOCK_R, BLOCK_Q
    )