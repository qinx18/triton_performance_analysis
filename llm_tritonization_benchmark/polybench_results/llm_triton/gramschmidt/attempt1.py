import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(
    A_ptr, Q_ptr, R_ptr,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    # Sequential k loop (outer dimension)
    for k in range(N):
        # Phase 1: Compute norm of column k
        nrm = 0.0
        m_offsets = tl.arange(0, BLOCK_M)
        
        # Process column k in blocks
        for m_start in range(0, M, BLOCK_M):
            current_m_offsets = m_start + m_offsets
            mask = current_m_offsets < M
            
            # Load A[i, k] values
            a_idx = current_m_offsets * N + k
            a_vals = tl.load(A_ptr + a_idx, mask=mask, other=0.0)
            
            # Accumulate squared values
            nrm += tl.sum(a_vals * a_vals)
        
        # Compute R[k, k] = sqrt(nrm)
        r_kk = tl.sqrt(nrm)
        r_kk_idx = k * N + k
        tl.store(R_ptr + r_kk_idx, r_kk)
        
        # Phase 2: Compute Q[:, k] = A[:, k] / R[k, k]
        for m_start in range(0, M, BLOCK_M):
            current_m_offsets = m_start + m_offsets
            mask = current_m_offsets < M
            
            # Load A[i, k] and compute Q[i, k]
            a_idx = current_m_offsets * N + k
            a_vals = tl.load(A_ptr + a_idx, mask=mask, other=0.0)
            q_vals = a_vals / r_kk
            
            # Store Q[i, k]
            q_idx = current_m_offsets * N + k
            tl.store(Q_ptr + q_idx, q_vals, mask=mask)
        
        # Phase 3: Process remaining columns j > k
        for j in range(k + 1, N):
            # Compute R[k, j] = sum(Q[i, k] * A[i, j])
            r_kj = 0.0
            
            for m_start in range(0, M, BLOCK_M):
                current_m_offsets = m_start + m_offsets
                mask = current_m_offsets < M
                
                # Load Q[i, k] and A[i, j]
                q_idx = current_m_offsets * N + k
                a_idx = current_m_offsets * N + j
                
                q_vals = tl.load(Q_ptr + q_idx, mask=mask, other=0.0)
                a_vals = tl.load(A_ptr + a_idx, mask=mask, other=0.0)
                
                # Accumulate dot product
                r_kj += tl.sum(q_vals * a_vals)
            
            # Store R[k, j]
            r_idx = k * N + j
            tl.store(R_ptr + r_idx, r_kj)
            
            # Update A[:, j] = A[:, j] - Q[:, k] * R[k, j]
            for m_start in range(0, M, BLOCK_M):
                current_m_offsets = m_start + m_offsets
                mask = current_m_offsets < M
                
                # Load Q[i, k] and A[i, j]
                q_idx = current_m_offsets * N + k
                a_idx = current_m_offsets * N + j
                
                q_vals = tl.load(Q_ptr + q_idx, mask=mask, other=0.0)
                a_vals = tl.load(A_ptr + a_idx, mask=mask, other=0.0)
                
                # Update A[i, j]
                new_a_vals = a_vals - q_vals * r_kj
                tl.store(A_ptr + a_idx, new_a_vals, mask=mask)

def gramschmidt_triton(A, Q, R, M, N):
    # Use block size of 64 for M dimension
    BLOCK_M = 64
    
    # Single kernel launch with grid=(1,) - everything sequential in one CTA
    grid = (1,)
    
    gramschmidt_kernel[grid](
        A, Q, R,
        M, N,
        BLOCK_M
    )