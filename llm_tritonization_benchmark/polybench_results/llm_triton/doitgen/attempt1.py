import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A, C4, sum, NP, NQ, NR, BLOCK_R: tl.constexpr, BLOCK_Q: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    r_start = pid_r * BLOCK_R
    q_start = pid_q * BLOCK_Q
    
    r_offsets = tl.arange(0, BLOCK_R)
    q_offsets = tl.arange(0, BLOCK_Q)
    p_offsets = tl.arange(0, 30)  # NP = 30
    s_offsets = tl.arange(0, 30)  # NP = 30
    
    r_mask = (r_start + r_offsets) < NR
    q_mask = (q_start + q_offsets) < NQ
    
    for r_idx in range(BLOCK_R):
        r = r_start + r_idx
        if r >= NR:
            break
            
        for q_idx in range(BLOCK_Q):
            q = q_start + q_idx
            if q >= NQ:
                break
                
            # Initialize sum array
            for p in range(30):  # NP = 30
                sum_val = 0.0
                
                # Compute sum[p] = sum_s(A[r][q][s] * C4[s][p])
                for s in range(30):  # NP = 30
                    a_idx = r * (NQ * 30) + q * 30 + s  # A[r][q][s]
                    c4_idx = s * 30 + p  # C4[s][p]
                    
                    a_val = tl.load(A + a_idx)
                    c4_val = tl.load(C4 + c4_idx)
                    sum_val += a_val * c4_val
                
                # Store to sum array
                sum_idx = p
                tl.store(sum + sum_idx, sum_val)
            
            # Copy sum back to A[r][q][:]
            for p in range(30):  # NP = 30
                sum_idx = p
                a_idx = r * (NQ * 30) + q * 30 + p  # A[r][q][p]
                
                sum_val = tl.load(sum + sum_idx)
                tl.store(A + a_idx, sum_val)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    BLOCK_R = 4
    BLOCK_Q = 4
    
    grid_r = triton.cdiv(NR, BLOCK_R)
    grid_q = triton.cdiv(NQ, BLOCK_Q)
    
    doitgen_kernel[(grid_r, grid_q)](
        A, C4, sum, NP, NQ, NR, BLOCK_R, BLOCK_Q
    )