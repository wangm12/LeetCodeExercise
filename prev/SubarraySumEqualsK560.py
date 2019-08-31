import collections
# Subarray Sum Equals K

# We cannot use sliding window in this question is because there
# might exist negative number

def subarraySum(nums, k):
    # edge case
    if not nums:
        return 0

    dic = collections.defaultdict(int)
    dic[0] = 1
    output = 0
    currentSum = 0
    for n in nums:
        currentSum += n
        if currentSum - k in dic:
            output += dic[currentSum-k]
        dic[currentSum] += 1
    return output


def OnlyPositive(nums, k):
    if not nums:
        return 0
    left, right = -1, -1
    totalSum = 0
    output = 0
    while right < len(nums):
        if totalSum == k:
            output += 1
            right += 1
            if right < len(nums):
                totalSum += nums[right]
            else:
                break
        elif totalSum < k:
            right += 1
            if right < len(nums):
                totalSum += nums[right]
            else:
                break
        else:
            while right > left and totalSum > k:
                left += 1
                totalSum -= nums[left]
    return output

test = [1,2,3,1,2]
print OnlyPositive(test, 3)
