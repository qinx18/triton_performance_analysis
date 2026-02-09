import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A, C4, NP, NQ, NR):
    # Get program IDs for parallelization
    pid_r = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    r = pid_r
    q = pid_q
    
    if r >= NR or q >= NQ:
        return
    
    # Process all p values sequentially
    for p in range(NP):
        # Initialize sum = 0.0
        sum_val = 0.0
        
        # Accumulate over s dimension
        for s in range(NP):
            # Load A[r][q][s]
            a_idx = r * (NQ * NP) + q * NP + s
            a_val = tl.load(A + a_idx)
            
            # Load C4[s][p]
            c4_idx = s * NP + p
            c4_val = tl.load(C4 + c4_idx)
            
            # Compute sum += A[r][q][s] * C4[s][p]
            sum_val += a_val * c4_val
        
        # Store sum back to A[r][q][p]
        a_out_idx = r * (NQ * NP) + q * NP + p
        tl.store(A + a_out_idx, sum_val)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Ensure tensors are contiguous and on GPU
    A = A.contiguous()
    C4 = C4.contiguous()
    
    # Launch configuration - parallelize over r and q dimensions
    grid = (NR, NQ)
    doitgen_kernel[grid](
        A, C4,
        NP, NQ, NR
    )