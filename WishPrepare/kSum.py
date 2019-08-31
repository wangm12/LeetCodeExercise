# K sum
def KSum(k, arr):
    output = []
    arr.sort()
    def helper(n, currentSum, index, current):
        if n == 2:
            left = index
            right = len(arr) - 1
            while left < right:
                if arr[left] + arr[right] + currentSum == 0:
                    output.append(current+[arr[left], arr[right]])
                    while left < right and arr[left] == arr[left+1]:
                        left += 1
                    left += 1
                    while right > left and arr[right] == arr[right-1]:
                        right -= 1
                    right -= 1
                elif arr[left] + arr[right] + currentSum < 0:
                    while left < right and arr[left] == arr[left+1]:
                        left += 1
                    left += 1
                else:
                    while right > left and arr[right] == arr[right-1]:
                        right -= 1
                    right -= 1
        else:
            for i in range (index, len(arr)-k+1):
                if i > index and arr[i] == arr[i-1]:
                    continue
                helper(n-1, currentSum+arr[i], i+1, current+[arr[i]])
    helper(k, 0, 0, [])
    return output
test = [1, 0, -1, 0, -2, 2]
print KSum(4, test)
