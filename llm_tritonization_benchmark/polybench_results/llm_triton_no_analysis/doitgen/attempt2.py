import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A_ptr, C4_ptr, sum_ptr, NP: tl.constexpr, NQ: tl.constexpr, NR: tl.constexpr):
    # Get program ID for parallelization over r and q dimensions
    pid = tl.program_id(0)
    r = pid // NQ
    q = pid % NQ
    
    # Early exit if out of bounds
    if r >= NR:
        return
    if q >= NQ:
        return
    
    # Compute matrix multiplication: sum[p] = sum_s(A[r,q,s] * C4[s,p])
    for p in range(NP):
        sum_val = 0.0
        for s in range(NP):
            # A[r][q][s] with linear indexing
            a_idx = r * (NQ * NP) + q * NP + s
            a_val = tl.load(A_ptr + a_idx)
            
            # C4[s][p] with linear indexing  
            c4_idx = s * NP + p
            c4_val = tl.load(C4_ptr + c4_idx)
            
            sum_val += a_val * c4_val
        
        # Store to sum[p] for this thread
        sum_offset = tl.program_id(0) * NP + p
        tl.store(sum_ptr + sum_offset, sum_val)
    
    # Copy sum back to A[r][q][:]
    for p in range(NP):
        sum_offset = tl.program_id(0) * NP + p
        sum_val = tl.load(sum_ptr + sum_offset)
        a_idx = r * (NQ * NP) + q * NP + p
        tl.store(A_ptr + a_idx, sum_val)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Launch kernel with one thread per (r,q) pair
    grid = (NR * NQ,)
    
    doitgen_kernel[grid](
        A, C4, sum,
        NP, NQ, NR
    )