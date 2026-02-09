import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(_fict_ptr, ex_ptr, ey_ptr, hz_ptr, 
                   NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr):
    
    # Time loop - must be sequential in FDTD
    for t in range(TMAX):
        # Step 1: Set boundary condition ey[0][j] = _fict_[t]
        fict_val = tl.load(_fict_ptr + t)
        for j in range(NY):
            ey_idx = 0 * NY + j  # row 0, column j
            tl.store(ey_ptr + ey_idx, fict_val)
        
        # Step 2: Update ey[i][j] for i=1 to NX-1
        for i in range(1, NX):
            for j in range(NY):
                ey_idx = i * NY + j
                hz_idx = i * NY + j
                hz_prev_idx = (i - 1) * NY + j
                
                ey_val = tl.load(ey_ptr + ey_idx)
                hz_val = tl.load(hz_ptr + hz_idx)
                hz_prev = tl.load(hz_ptr + hz_prev_idx)
                
                new_ey = ey_val - 0.5 * (hz_val - hz_prev)
                tl.store(ey_ptr + ey_idx, new_ey)
        
        # Step 3: Update ex[i][j] for j=1 to NY-1
        for i in range(NX):
            for j in range(1, NY):
                ex_idx = i * NY + j
                hz_idx = i * NY + j
                hz_prev_idx = i * NY + (j - 1)
                
                ex_val = tl.load(ex_ptr + ex_idx)
                hz_val = tl.load(hz_ptr + hz_idx)
                hz_prev = tl.load(hz_ptr + hz_prev_idx)
                
                new_ex = ex_val - 0.5 * (hz_val - hz_prev)
                tl.store(ex_ptr + ex_idx, new_ex)
        
        # Step 4: Update hz[i][j] for i=0 to NX-2, j=0 to NY-2
        for i in range(NX - 1):
            for j in range(NY - 1):
                hz_idx = i * NY + j
                ex_idx = i * NY + j
                ex_next_idx = i * NY + (j + 1)
                ey_idx = i * NY + j
                ey_next_idx = (i + 1) * NY + j
                
                hz_val = tl.load(hz_ptr + hz_idx)
                ex_val = tl.load(ex_ptr + ex_idx)
                ex_next = tl.load(ex_ptr + ex_next_idx)
                ey_val = tl.load(ey_ptr + ey_idx)
                ey_next = tl.load(ey_ptr + ey_next_idx)
                
                new_hz = hz_val - 0.7 * ((ex_next - ex_val) + (ey_next - ey_val))
                tl.store(hz_ptr + hz_idx, new_hz)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    fdtd_2d_kernel[(1,)](
        _fict_, ex, ey, hz,
        NX=NX, NY=NY, TMAX=TMAX
    )