import triton
import triton.language as tl
import torch

@triton.jit
def k3mm_kernel(
    A, B, C, D, E, F, G,
    NI, NJ, NK, NL, NM,
    BLOCK_SIZE: tl.constexpr
):
    # Get thread/block IDs
    pid = tl.program_id(0)
    
    # E := A*B
    if pid == 0:
        for i in range(NI):
            for j in range(NJ):
                accumulator = 0.0
                for k in range(NK):
                    a_val = tl.load(A + i * NK + k)
                    b_val = tl.load(B + k * NJ + j)
                    accumulator += a_val * b_val
                tl.store(E + i * NJ + j, accumulator)
        
        # F := C*D
        for i in range(NJ):
            for j in range(NL):
                accumulator = 0.0
                for k in range(NM):
                    c_val = tl.load(C + i * NM + k)
                    d_val = tl.load(D + k * NL + j)
                    accumulator += c_val * d_val
                tl.store(F + i * NL + j, accumulator)
        
        # G := E*F
        for i in range(NI):
            for j in range(NL):
                accumulator = 0.0
                for k in range(NJ):
                    e_val = tl.load(E + i * NJ + k)
                    f_val = tl.load(F + k * NL + j)
                    accumulator += e_val * f_val
                tl.store(G + i * NL + j, accumulator)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_SIZE = 128
    
    grid = (1,)
    
    k3mm_kernel[grid](
        A, B, C, D, E, F, G,
        NI, NJ, NK, NL, NM,
        BLOCK_SIZE=BLOCK_SIZE
    )