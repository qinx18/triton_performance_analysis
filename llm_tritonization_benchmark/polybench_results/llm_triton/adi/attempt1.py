import triton
import triton.language as tl
import torch

@triton.jit
def adi_column_sweep_kernel(
    u_ptr, v_ptr, p_ptr, q_ptr,
    a, b, c, d, f,
    N: tl.constexpr
):
    i = tl.program_id(0) + 1  # Skip boundary, i in [1, N-2]
    
    # Boundary conditions
    tl.store(v_ptr + 0 * N + i, 1.0)
    tl.store(p_ptr + i * N + 0, 0.0)
    tl.store(q_ptr + i * N + 0, 1.0)
    
    # Forward sweep
    for j in range(1, N-1):
        p_prev = tl.load(p_ptr + i * N + (j-1))
        q_prev = tl.load(q_ptr + i * N + (j-1))
        
        u_jm1_i = tl.load(u_ptr + j * N + (i-1))
        u_j_i = tl.load(u_ptr + j * N + i)
        u_jp1_i = tl.load(u_ptr + j * N + (i+1))
        
        p_val = -c / (a * p_prev + b)
        q_val = (-d * u_jm1_i + (1.0 + 2.0 * d) * u_j_i - f * u_jp1_i - a * q_prev) / (a * p_prev + b)
        
        tl.store(p_ptr + i * N + j, p_val)
        tl.store(q_ptr + i * N + j, q_val)
    
    # Boundary condition
    tl.store(v_ptr + (N-1) * N + i, 1.0)
    
    # Backward sweep
    for j in range(N-2, 0, -1):
        p_val = tl.load(p_ptr + i * N + j)
        q_val = tl.load(q_ptr + i * N + j)
        v_next = tl.load(v_ptr + (j+1) * N + i)
        
        v_val = p_val * v_next + q_val
        tl.store(v_ptr + j * N + i, v_val)

@triton.jit
def adi_row_sweep_kernel(
    u_ptr, v_ptr, p_ptr, q_ptr,
    a, b, c, d, e, f,
    N: tl.constexpr
):
    i = tl.program_id(0) + 1  # Skip boundary, i in [1, N-2]
    
    # Boundary conditions
    tl.store(u_ptr + i * N + 0, 1.0)
    tl.store(p_ptr + i * N + 0, 0.0)
    tl.store(q_ptr + i * N + 0, 1.0)
    
    # Forward sweep
    for j in range(1, N-1):
        p_prev = tl.load(p_ptr + i * N + (j-1))
        q_prev = tl.load(q_ptr + i * N + (j-1))
        
        v_im1_j = tl.load(v_ptr + (i-1) * N + j)
        v_i_j = tl.load(v_ptr + i * N + j)
        v_ip1_j = tl.load(v_ptr + (i+1) * N + j)
        
        p_val = -f / (d * p_prev + e)
        q_val = (-a * v_im1_j + (1.0 + 2.0 * a) * v_i_j - c * v_ip1_j - d * q_prev) / (d * p_prev + e)
        
        tl.store(p_ptr + i * N + j, p_val)
        tl.store(q_ptr + i * N + j, q_val)
    
    # Boundary condition
    tl.store(u_ptr + i * N + (N-1), 1.0)
    
    # Backward sweep
    for j in range(N-2, 0, -1):
        p_val = tl.load(p_ptr + i * N + j)
        q_val = tl.load(q_ptr + i * N + j)
        u_next = tl.load(u_ptr + i * N + (j+1))
        
        u_val = p_val * u_next + q_val
        tl.store(u_ptr + i * N + j, u_val)

def adi_triton(p, q, u, v, N, TSTEPS):
    # Compute constants
    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)
    
    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d
    
    # Time stepping loop
    for t in range(1, TSTEPS + 1):
        # Column sweep - parallelize over i
        grid = (N - 2,)
        adi_column_sweep_kernel[grid](
            u, v, p, q,
            a, b, c, d, f,
            N
        )
        
        # Row sweep - parallelize over i
        adi_row_sweep_kernel[grid](
            u, v, p, q,
            a, b, c, d, e, f,
            N
        )