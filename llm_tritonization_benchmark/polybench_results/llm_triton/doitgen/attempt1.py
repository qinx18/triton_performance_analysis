import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A_out, A_in, C4, NP: tl.constexpr, NQ: tl.constexpr, NR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    
    # Decode r and q from pid
    r = pid // NQ
    q = pid % NQ
    
    # Vectorize over p dimension
    p_offsets = tl.arange(0, BLOCK)
    p_mask = p_offsets < NP
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    
    # Compute matrix-vector product: sum[p] = A[r,q,s] * C4[s,p] for all s
    for s in range(NP):
        # Load scalar A[r,q,s] from input copy
        a_idx = r * (NQ * NP) + q * NP + s
        a_val = tl.load(A_in + a_idx)
        
        # Load vector C4[s,p] 
        c4_base = s * NP
        c4_vals = tl.load(C4 + c4_base + p_offsets, mask=p_mask)
        
        # Accumulate
        acc += a_val * c4_vals
    
    # Store result back to A[r,q,p]
    a_out_base = r * (NQ * NP) + q * NP
    tl.store(A_out + a_out_base + p_offsets, acc, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Clone A for read-only access to avoid read-after-write hazards
    A_copy = A.clone()
    
    BLOCK = min(triton.next_power_of_2(NP), 128)
    grid = (NR * NQ,)
    
    doitgen_kernel[grid](A, A_copy, C4, NP, NQ, NR, BLOCK)