import collections
# shortestSubarray Sum greater than target
# All positive Number
def solution(arr, target):
    if not arr or sum(arr) < target: return []
    left, right, currentSum = 0, 0, 0
    outputLength = float("inf")
    output = []
    for right in range (len(arr)):
        currentSum += arr[right]
        while currentSum >= target:
            if right - left + 1 < outputLength:
                outputLength = right - left + 1
                output = arr[left:right+1]
            currentSum -= arr[left]
            left += 1
    return output

# test = [2,3,1,2,2,3]
# test2 = [7,2,3,1,2,4,3]
# test3 = [3,3,3,2,3,1,2,4,3]
# print solution(test3,7)


# Includes Negative Number
def solution2(arr, target):
    if not arr: return []
    totalSum = [0]
    for num in arr: totalSum.append(num+totalSum[-1])
    dq = collections.deque()
    output = float("inf")
    for i, v in enumerate(totalSum):
        while dq and totalSum[dq[-1]] > v:
            dq.pop()
        while dq and v - totalSum[dq[0]] > target:
            output = min(output, i - dq.popleft())
        dq.append(i)
    if output == float("inf"): return -1
    return output
print solution2([-2,1,2], 3)
