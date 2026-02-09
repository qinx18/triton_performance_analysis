import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A, C4, sum_ptr, NP, NQ, NR):
    # Get program IDs for parallelization
    pid_r = tl.program_id(0)
    pid_q = tl.program_id(1)
    
    r = pid_r
    q = pid_q
    
    if r >= NR or q >= NQ:
        return
    
    # Process all p values sequentially
    for p in range(NP):
        # Initialize sum[p] = 0.0
        sum_val = 0.0
        
        # Accumulate over s dimension in blocks
        s_offsets = tl.arange(0, 32)
        for s_block in range(0, NP, 32):
            s_indices = s_block + s_offsets
            s_mask = s_indices < NP
            
            # Load A[r][q][s] values
            a_idx = r * (NQ * NP) + q * NP + s_indices
            a_vals = tl.load(A + a_idx, mask=s_mask, other=0.0)
            
            # Load C4[s][p] values
            c4_idx = s_indices * NP + p
            c4_vals = tl.load(C4 + c4_idx, mask=s_mask, other=0.0)
            
            # Compute sum += A[r][q][s] * C4[s][p]
            products = a_vals * c4_vals
            sum_val += tl.sum(products)
        
        # Store sum[p] back to A[r][q][p]
        a_out_idx = r * (NQ * NP) + q * NP + p
        tl.store(A + a_out_idx, sum_val)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Ensure tensors are contiguous and on GPU
    A = A.contiguous()
    C4 = C4.contiguous()
    sum = sum.contiguous()
    
    # Launch configuration - parallelize over r and q dimensions
    grid = (NR, NQ)
    doitgen_kernel[grid](
        A, C4, sum,
        NP, NQ, NR
    )