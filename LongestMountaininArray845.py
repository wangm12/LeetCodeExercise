# Leetcode

def longestMountain(self, A):
"""
:type A: List[int]
:rtype: int
"""
# edge case
if not A or len(A) < 3: return 0

# 3 pass
up = [0] * len(A)
down = [0] * len(A)
for i in range (1, len(A)):
    if A[i] > A[i-1]:
        up[i] = up[i-1]+1
for i in range (len(A)-2, -1, -1):
    if A[i] > A[i+1]:
        down[i] = down[i+1]+1
output = 0
for i in range (len(A)):
    if up[i] and down[i]:
        output = max(output, up[i]+down[i]+1)
if output < 3: return 0
return output

# 1 pass
