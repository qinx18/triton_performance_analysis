import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A_out, A_in, C4, NP: tl.constexpr, NQ: tl.constexpr, NR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Decode r and q from pid
    r = pid // NQ
    q = pid % NQ
    
    # Vector offsets for p dimension
    p_offsets = tl.arange(0, BLOCK)
    mask = p_offsets < NP
    
    # Compute matrix multiplication: sum[p] = A[r,q,s] * C4[s,p] for all s
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    
    for s in range(NP):
        # Load scalar A[r,q,s]
        a_idx = r * (NQ * NP) + q * NP + s
        a_val = tl.load(A_in + a_idx)
        
        # Load vector C4[s,p]
        c4_idx = s * NP + p_offsets
        c4_vec = tl.load(C4 + c4_idx, mask=mask)
        
        # Accumulate
        acc += a_val * c4_vec
    
    # Store result back to A[r,q,p]
    a_out_idx = r * (NQ * NP) + q * NP + p_offsets
    tl.store(A_out + a_out_idx, acc, mask=mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    A_copy = A.clone()
    BLOCK = triton.next_power_of_2(NP)
    grid = (NR * NQ,)
    
    doitgen_kernel[grid](A, A_copy, C4, NP, NQ, NR, BLOCK)