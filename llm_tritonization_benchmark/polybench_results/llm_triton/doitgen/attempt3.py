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
        
    # Define offset arrays once
    p_offsets = tl.arange(0, 32)
    s_offsets = tl.arange(0, 32)
    
    # Process p dimension in blocks
    for p_block in range(0, NP, 32):
        p_indices = p_block + p_offsets
        p_mask = p_indices < NP
        
        # Initialize sum values to zero
        sum_vals = tl.zeros([32], dtype=tl.float32)
        
        # Accumulate over s dimension
        for s_block in range(0, NP, 32):
            s_indices = s_block + s_offsets
            s_mask = s_indices < NP
            
            # Load A[r][q][s] values
            a_idx = r * (NQ * NP) + q * NP + s_indices
            a_vals = tl.load(A + a_idx, mask=s_mask, other=0.0)
            
            # For each p in current block
            p_local = 0
            while p_local < 32:
                p_global = p_block + p_local
                if p_global < NP:
                    # Load C4[s][p] values for this p
                    c4_idx = s_indices * NP + p_global
                    c4_vals = tl.load(C4 + c4_idx, mask=s_mask, other=0.0)
                    
                    # Compute sum[p] += A[r][q][s] * C4[s][p]
                    product = a_vals * c4_vals
                    local_sum = tl.sum(product)
                    
                    # Add to appropriate position in sum_vals
                    current_sum = sum_vals + (0 * 32)  # Load current value
                    updated_sum = tl.where(p_offsets == p_local, 
                                         tl.broadcast_to(local_sum, [32]) + current_sum, 
                                         current_sum)
                    sum_vals = updated_sum
                p_local = p_local + 1
        
        # Store final sum values back to A[r][q][p]
        a_out_idx = r * (NQ * NP) + q * NP + p_indices
        tl.store(A + a_out_idx, sum_vals, mask=p_mask)

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