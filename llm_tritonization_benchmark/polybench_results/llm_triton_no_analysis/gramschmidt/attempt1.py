import torch
import triton
import triton.language as tl

@triton.jit
def gramschmidt_kernel(A_ptr, Q_ptr, R_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    for k in range(N):
        # Compute norm squared
        nrm = 0.0
        
        # Process column k in blocks for norm computation
        for block_start in range(0, M, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            i_offsets = block_start + offsets
            mask = i_offsets < M
            
            # Load A[i][k] values
            a_vals = tl.load(A_ptr + i_offsets * N + k, mask=mask, other=0.0)
            nrm += tl.sum(a_vals * a_vals)
        
        # Compute R[k][k] = sqrt(nrm)
        r_kk = tl.sqrt(nrm)
        tl.store(R_ptr + k * N + k, r_kk)
        
        # Compute Q[i][k] = A[i][k] / R[k][k] for all i
        for block_start in range(0, M, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            i_offsets = block_start + offsets
            mask = i_offsets < M
            
            # Load A[i][k] values
            a_vals = tl.load(A_ptr + i_offsets * N + k, mask=mask, other=0.0)
            q_vals = a_vals / r_kk
            tl.store(Q_ptr + i_offsets * N + k, q_vals, mask=mask)
        
        # Process remaining columns j = k+1 to N-1
        for j in range(k + 1, N):
            # Compute R[k][j] = sum(Q[i][k] * A[i][j])
            r_kj = 0.0
            
            for block_start in range(0, M, BLOCK_SIZE):
                offsets = tl.arange(0, BLOCK_SIZE)
                i_offsets = block_start + offsets
                mask = i_offsets < M
                
                q_vals = tl.load(Q_ptr + i_offsets * N + k, mask=mask, other=0.0)
                a_vals = tl.load(A_ptr + i_offsets * N + j, mask=mask, other=0.0)
                r_kj += tl.sum(q_vals * a_vals)
            
            tl.store(R_ptr + k * N + j, r_kj)
            
            # Update A[i][j] = A[i][j] - Q[i][k] * R[k][j] for all i
            for block_start in range(0, M, BLOCK_SIZE):
                offsets = tl.arange(0, BLOCK_SIZE)
                i_offsets = block_start + offsets
                mask = i_offsets < M
                
                q_vals = tl.load(Q_ptr + i_offsets * N + k, mask=mask, other=0.0)
                a_vals = tl.load(A_ptr + i_offsets * N + j, mask=mask, other=0.0)
                new_a_vals = a_vals - q_vals * r_kj
                tl.store(A_ptr + i_offsets * N + j, new_a_vals, mask=mask)

def gramschmidt_triton(A, Q, R, M, N):
    BLOCK_SIZE = 64
    
    # Launch single thread since algorithm is inherently sequential in k
    grid = (1,)
    
    gramschmidt_kernel[grid](
        A, Q, R,
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )