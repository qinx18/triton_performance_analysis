#!/usr/bin/env python3
"""
Debug the diagonal calculation logic
"""

M, N = 5, 5

# Number of diagonals to process
max_diag = M + N - 4
print(f"M={M}, N={N}")
print(f"max_diag = {max_diag}")
print()

for diag in range(max_diag + 1):
    # Current calculation
    num_elements = min(diag + 1, M - 1, N - 1 - max(0, diag - M + 2))

    print(f"Diagonal {diag}:")
    print(f"  num_elements (current formula) = {num_elements}")

    # Manual calculation of which elements should be in this diagonal
    elements = []
    for pid in range(100):  # Large number
        j = pid + 1
        i = diag + 1 - pid
        if j < M and i < N and i >= 1:
            elements.append((j, i))
        else:
            break

    print(f"  actual elements: {elements}")
    print(f"  actual count: {len(elements)}")
    print()
