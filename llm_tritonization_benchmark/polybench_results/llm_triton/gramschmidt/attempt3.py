import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(A, Q, R, M, N, BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    k = block_id
    
    if k >= N:
        return
    
    # Compute nrm = sum(A[i][k] * A[i][k]) for all i
    nrm = 0.0
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for block_start in range(0, M, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < M
        
        a_vals = tl.load(A + current_offsets * N + k, mask=mask, other=0.0)
        nrm += tl.sum(a_vals * a_vals, axis=0)
    
    # R[k][k] = sqrt(nrm)
    r_kk = tl.sqrt(nrm)
    tl.store(R + k * N + k, r_kk)
    
    # Q[i][k] = A[i][k] / R[k][k] for all i
    for block_start in range(0, M, BLOCK_SIZE):
        current_offsets = block_start + offsets
        mask = current_offsets < M
        
        a_vals = tl.load(A + current_offsets * N + k, mask=mask, other=0.0)
        q_vals = a_vals / r_kk
        tl.store(Q + current_offsets * N + k, q_vals, mask=mask)
    
    # For each j > k
    for j in range(k + 1, N):
        # R[k][j] = sum(Q[i][k] * A[i][j]) for all i
        r_kj = 0.0
        
        for block_start in range(0, M, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < M
            
            q_vals = tl.load(Q + current_offsets * N + k, mask=mask, other=0.0)
            a_vals = tl.load(A + current_offsets * N + j, mask=mask, other=0.0)
            r_kj += tl.sum(q_vals * a_vals, axis=0)
        
        tl.store(R + k * N + j, r_kj)
        
        # A[i][j] = A[i][j] - Q[i][k] * R[k][j] for all i
        for block_start in range(0, M, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < M
            
            a_vals = tl.load(A + current_offsets * N + j, mask=mask, other=0.0)
            q_vals = tl.load(Q + current_offsets * N + k, mask=mask, other=0.0)
            new_a_vals = a_vals - q_vals * r_kj
            tl.store(A + current_offsets * N + j, new_a_vals, mask=mask)

def gramschmidt_triton(A, Q, R, M, N):
    BLOCK_SIZE = 64
    
    for k in range(N):
        grid = (1,)
        gramschmidt_kernel[grid](A, Q, R, M, N, BLOCK_SIZE)