import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(A, Q, R, M, N, k):
    # Compute nrm = sum(A[i][k] * A[i][k]) for all i
    nrm = 0.0
    for i in range(M):
        a_val = tl.load(A + i * N + k)
        nrm += a_val * a_val
    
    # R[k][k] = sqrt(nrm)
    r_kk = tl.sqrt(nrm)
    tl.store(R + k * N + k, r_kk)
    
    # Q[i][k] = A[i][k] / R[k][k] for all i
    for i in range(M):
        a_val = tl.load(A + i * N + k)
        q_val = a_val / r_kk
        tl.store(Q + i * N + k, q_val)
    
    # For each j > k
    for j in range(k + 1, N):
        # R[k][j] = sum(Q[i][k] * A[i][j]) for all i
        r_kj = 0.0
        for i in range(M):
            q_val = tl.load(Q + i * N + k)
            a_val = tl.load(A + i * N + j)
            r_kj += q_val * a_val
        tl.store(R + k * N + j, r_kj)
        
        # A[i][j] = A[i][j] - Q[i][k] * R[k][j] for all i
        for i in range(M):
            a_val = tl.load(A + i * N + j)
            q_val = tl.load(Q + i * N + k)
            new_a_val = a_val - q_val * r_kj
            tl.store(A + i * N + j, new_a_val)

def gramschmidt_triton(A, Q, R, M, N):
    for k in range(N):
        grid = (1,)
        gramschmidt_kernel[grid](A, Q, R, M, N, k)