import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel(_fict_ptr, ex_ptr, ey_ptr, hz_ptr, 
                   NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr):
    
    for t in range(TMAX):
        # Phase 1: Set boundary conditions for ey[0][j]
        for j in range(NY):
            fict_val = tl.load(_fict_ptr + t)
            tl.store(ey_ptr + j, fict_val)
        
        # Phase 2: Update ey[i][j] for i >= 1
        for i in range(1, NX):
            for j in range(NY):
                ey_offset = i * NY + j
                hz_offset = i * NY + j
                hz_prev_offset = (i - 1) * NY + j
                
                ey_val = tl.load(ey_ptr + ey_offset)
                hz_val = tl.load(hz_ptr + hz_offset)
                hz_prev_val = tl.load(hz_ptr + hz_prev_offset)
                
                new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ey_ptr + ey_offset, new_ey)
        
        # Phase 3: Update ex[i][j] for j >= 1
        for i in range(NX):
            for j in range(1, NY):
                ex_offset = i * NY + j
                hz_offset = i * NY + j
                hz_prev_offset = i * NY + (j - 1)
                
                ex_val = tl.load(ex_ptr + ex_offset)
                hz_val = tl.load(hz_ptr + hz_offset)
                hz_prev_val = tl.load(hz_ptr + hz_prev_offset)
                
                new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
                tl.store(ex_ptr + ex_offset, new_ex)
        
        # Phase 4: Update hz[i][j] for i < NX-1, j < NY-1
        for i in range(NX - 1):
            for j in range(NY - 1):
                hz_offset = i * NY + j
                ex_next_offset = i * NY + (j + 1)
                ex_curr_offset = i * NY + j
                ey_next_offset = (i + 1) * NY + j
                ey_curr_offset = i * NY + j
                
                hz_val = tl.load(hz_ptr + hz_offset)
                ex_next_val = tl.load(ex_ptr + ex_next_offset)
                ex_curr_val = tl.load(ex_ptr + ex_curr_offset)
                ey_next_val = tl.load(ey_ptr + ey_next_offset)
                ey_curr_val = tl.load(ey_ptr + ey_curr_offset)
                
                new_hz = hz_val - 0.7 * ((ex_next_val - ex_curr_val) + 
                                        (ey_next_val - ey_curr_val))
                tl.store(hz_ptr + hz_offset, new_hz)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    grid = (1,)
    fdtd_2d_kernel[grid](_fict_, ex, ey, hz, NX, NY, TMAX)