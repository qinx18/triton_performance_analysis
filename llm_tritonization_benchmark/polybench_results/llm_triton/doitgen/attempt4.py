import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A, C4, sum, NP, NQ, NR, BLOCK_R: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_P: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    r_start = pid_r * BLOCK_R
    q_start = pid_q * BLOCK_Q
    
    p_offsets = tl.arange(0, BLOCK_P)
    
    for r_idx in tl.static_range(BLOCK_R):
        r = r_start + r_idx
        r_is_valid = r < NR
        
        for q_idx in tl.static_range(BLOCK_Q):
            q = q_start + q_idx
            q_is_valid = q < NQ
            
            valid = r_is_valid & q_is_valid
            
            if valid:
                sum_vals = tl.zeros([BLOCK_P], dtype=tl.float32)
                
                for s in range(NP):
                    a_idx = r * (NQ * NP) + q * NP + s
                    a_val = tl.load(A + a_idx)
                    
                    c4_ptrs = C4 + s * NP + p_offsets
                    p_mask = p_offsets < NP
                    c4_vals = tl.load(c4_ptrs, mask=p_mask)
                    
                    sum_vals += a_val * c4_vals
                
                a_ptrs = A + r * (NQ * NP) + q * NP + p_offsets
                p_mask = p_offsets < NP
                tl.store(a_ptrs, sum_vals, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    BLOCK_R = 4
    BLOCK_Q = 4
    BLOCK_P = 32
    
    grid_r = triton.cdiv(NR, BLOCK_R)
    grid_q = triton.cdiv(NQ, BLOCK_Q)
    
    doitgen_kernel[(grid_r, grid_q)](
        A, C4, sum, NP, NQ, NR, BLOCK_R, BLOCK_Q, BLOCK_P
    )