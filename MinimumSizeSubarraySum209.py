def minSubArrayLen(s, nums):
    """
    :type s: int
    :type nums: List[int]
    :rtype: int
    """
    if not s or not nums:
        return 0

    left = 0
    right = 0
    total = 0
    output = float("inf")
    while right < len(nums):
        total += nums[right]
        while total >= s:
            output = min(output, right - left + 1)
            total -= nums[left]
            left += 1
        right += 1
    if output == float("inf"):
        return 0
    return output


print minSubArrayLen(7, [2,3,1,2,4,3])
