import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(A, Q, R, k: tl.constexpr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Step 1: Compute norm of column k
    nrm = 0.0
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_start in range(0, M, BLOCK_SIZE):
        current_i_offsets = i_start + i_offsets
        mask = current_i_offsets < M
        
        # Load A[i][k]
        a_vals = tl.load(A + current_i_offsets * N + k, mask=mask, other=0.0)
        nrm += tl.sum(a_vals * a_vals)
    
    # Step 2: Set R[k][k] = sqrt(nrm)
    rkk = tl.sqrt(nrm)
    tl.store(R + k * N + k, rkk)
    
    # Step 3: Compute Q[i][k] = A[i][k] / R[k][k]
    for i_start in range(0, M, BLOCK_SIZE):
        current_i_offsets = i_start + i_offsets
        mask = current_i_offsets < M
        
        a_vals = tl.load(A + current_i_offsets * N + k, mask=mask, other=0.0)
        q_vals = a_vals / rkk
        tl.store(Q + current_i_offsets * N + k, q_vals, mask=mask)
    
    # Step 4: Update remaining columns j > k
    for j in range(k + 1, N):
        # Compute R[k][j] = sum(Q[i][k] * A[i][j])
        rkj = 0.0
        
        for i_start in range(0, M, BLOCK_SIZE):
            current_i_offsets = i_start + i_offsets
            mask = current_i_offsets < M
            
            q_vals = tl.load(Q + current_i_offsets * N + k, mask=mask, other=0.0)
            a_vals = tl.load(A + current_i_offsets * N + j, mask=mask, other=0.0)
            rkj += tl.sum(q_vals * a_vals)
        
        # Store R[k][j]
        tl.store(R + k * N + j, rkj)
        
        # Update A[i][j] = A[i][j] - Q[i][k] * R[k][j]
        for i_start in range(0, M, BLOCK_SIZE):
            current_i_offsets = i_start + i_offsets
            mask = current_i_offsets < M
            
            q_vals = tl.load(Q + current_i_offsets * N + k, mask=mask, other=0.0)
            a_vals = tl.load(A + current_i_offsets * N + j, mask=mask, other=0.0)
            new_a_vals = a_vals - q_vals * rkj
            tl.store(A + current_i_offsets * N + j, new_a_vals, mask=mask)

def gramschmidt_triton(A, Q, R, M, N):
    BLOCK_SIZE = 64
    
    # Initialize R to zero and copy A to work matrix
    R.zero_()
    A_work = A.clone()
    
    # Sequential execution for each column
    for k in range(N):
        grid = (1,)
        
        gramschmidt_kernel[grid](
            A_work, Q, R, 
            k, M, N, 
            BLOCK_SIZE=BLOCK_SIZE
        )