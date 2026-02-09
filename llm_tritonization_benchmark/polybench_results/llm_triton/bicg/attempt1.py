import triton
import triton.language as tl
import torch

@triton.jit
def bicg_kernel(A_ptr, p_ptr, q_ptr, r_ptr, s_ptr, M, N, 
                A_stride0, A_stride1,
                BLOCK_SIZE: tl.constexpr):
    
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which row this program handles
    i = pid
    
    if i < N:
        # Initialize q[i] = 0.0
        q_offset = i
        tl.store(q_ptr + q_offset, 0.0)
        
        # Load r[i]
        r_i = tl.load(r_ptr + i)
        
        # Initialize q_sum for this row
        q_sum = 0.0
        
        # Process blocks of M
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, M, BLOCK_SIZE):
            current_j_offsets = j_start + j_offsets
            j_mask = current_j_offsets < M
            
            # Load A[i][j] values
            A_offsets = i * A_stride0 + current_j_offsets * A_stride1
            A_vals = tl.load(A_ptr + A_offsets, mask=j_mask, other=0.0)
            
            # Load p[j] values
            p_vals = tl.load(p_ptr + current_j_offsets, mask=j_mask, other=0.0)
            
            # Load current s[j] values
            s_vals = tl.load(s_ptr + current_j_offsets, mask=j_mask, other=0.0)
            
            # Update s[j] = s[j] + r[i] * A[i][j]
            new_s_vals = s_vals + r_i * A_vals
            tl.store(s_ptr + current_j_offsets, new_s_vals, mask=j_mask)
            
            # Accumulate q[i] += A[i][j] * p[j]
            q_sum += tl.sum(A_vals * p_vals)
        
        # Store final q[i] value
        tl.store(q_ptr + q_offset, q_sum)

def bicg_triton(A, p, q, r, s, M, N):
    # Initialize s to zeros
    s.zero_()
    
    # Set up grid and block size
    BLOCK_SIZE = 64
    grid = (N,)
    
    # Launch kernel
    bicg_kernel[grid](
        A, p, q, r, s, M, N,
        A.stride(0), A.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )