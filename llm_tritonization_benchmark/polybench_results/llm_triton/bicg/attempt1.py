import triton
import triton.language as tl
import torch

@triton.jit
def bicg_kernel(A_ptr, p_ptr, q_ptr, r_ptr, s_ptr, M, N, 
                A_stride_0, A_stride_1,
                BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Initialize s to 0 (first loop)
    s_offsets = block_start + offsets
    s_mask = s_offsets < M
    tl.store(s_ptr + s_offsets, 0.0, mask=s_mask)
    
    # Main nested loops
    for i in range(N):
        # Initialize q[i] to 0
        if block_start == 0:
            tl.store(q_ptr + i, 0.0)
        
        # Inner loop over j (M dimension)
        r_i = tl.load(r_ptr + i)
        q_i = 0.0
        
        for j_start in range(0, M, BLOCK_SIZE):
            j_offsets = j_start + offsets
            j_mask = j_offsets < M
            
            # Load A[i][j]
            A_indices = i * A_stride_0 + j_offsets * A_stride_1
            A_vals = tl.load(A_ptr + A_indices, mask=j_mask, other=0.0)
            
            # Load p[j]
            p_vals = tl.load(p_ptr + j_offsets, mask=j_mask, other=0.0)
            
            # Load current s[j] values
            s_vals = tl.load(s_ptr + j_offsets, mask=j_mask, other=0.0)
            
            # Update s[j] = s[j] + r[i] * A[i][j]
            new_s_vals = s_vals + r_i * A_vals
            tl.store(s_ptr + j_offsets, new_s_vals, mask=j_mask)
            
            # Accumulate q[i] += A[i][j] * p[j]
            q_i += tl.sum(A_vals * p_vals, axis=0)
        
        # Update q[i] with accumulated value
        if block_start == 0:
            q_current = tl.load(q_ptr + i)
            tl.store(q_ptr + i, q_current + q_i)

def bicg_triton(A, p, q, r, s, M, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(M, BLOCK_SIZE),)
    
    bicg_kernel[grid](
        A, p, q, r, s, M, N,
        A.stride(0), A.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )