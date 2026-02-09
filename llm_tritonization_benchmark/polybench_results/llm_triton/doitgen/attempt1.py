import triton
import triton.language as tl
import torch

@triton.jit
def doitgen_kernel(A, C4, sum_ptr, NP, NQ, NR, r_start, num_r):
    # Get program ID for the q dimension
    pid_q = tl.program_id(0)
    
    # Process multiple r values in this kernel instance
    for r_offset in range(num_r):
        r = r_start + r_offset
        if r >= NR:
            break
            
        q = pid_q
        if q >= NQ:
            return
            
        # Define offset arrays once
        p_offsets = tl.arange(0, 32)  # Use power of 2 for efficiency
        s_offsets = tl.arange(0, 32)
        
        # Compute sum for this (r, q) pair
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
                
                # Compute outer product and accumulate
                for p_i in range(32):
                    if p_block + p_i >= NP:
                        break
                    p_actual = p_block + p_i
                    
                    # Load C4[s][p] values for this p
                    c4_idx = s_indices * NP + p_actual
                    c4_vals = tl.load(C4 + c4_idx, mask=s_mask, other=0.0)
                    
                    # Accumulate sum[p] += A[r][q][s] * C4[s][p]
                    product = a_vals * c4_vals
                    sum_vals = tl.where(p_offsets == p_i, sum_vals + tl.sum(product), sum_vals)
            
            # Store sum values and update A
            sum_idx = p_indices
            tl.store(sum_ptr + sum_idx, sum_vals, mask=p_mask)
            
            # Load sum values and store to A[r][q][p]
            final_sum = tl.load(sum_ptr + p_indices, mask=p_mask)
            a_out_idx = r * (NQ * NP) + q * NP + p_indices
            tl.store(A + a_out_idx, final_sum, mask=p_mask)

def doitgen_triton(A, C4, sum, NP, NQ, NR):
    # Ensure tensors are contiguous and on GPU
    A = A.contiguous()
    C4 = C4.contiguous()
    sum = sum.contiguous()
    
    # Launch configuration
    BLOCK_R = 4  # Process multiple r values per kernel
    
    # Launch kernels for different r ranges
    for r_start in range(0, NR, BLOCK_R):
        num_r = min(BLOCK_R, NR - r_start)
        
        grid = (NQ,)
        doitgen_kernel[grid](
            A, C4, sum, 
            NP, NQ, NR,
            r_start, num_r
        )