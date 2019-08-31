# Wish Prepare

# Sliding Window -> Longest subarray sum > k
# it doesn't
# def longestSubarraySumGreaterThanK(nums, k):
#     if not nums: return 0
import collections
import time

def numIslands(grid):
    def helper(i, j):
        if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == "1":
            grid[i][j] = "2"
            helper(i+1, j)
            helper(i-1, j)
            helper(i, j+1)
            helper(i, j-1)


    if not grid or not grid[0]:
        return 0

    island = 0
    for i in range (len(grid)):
        for j in range (len(grid[0])):
            if grid[i][j] == "1":
                island += 1
                helper(i, j)

    return island


# 128
def longestConsecutive(nums):
    if not nums: return 0

    s = set(nums)
    output = 0
    while s:
        current = s.pop()
        temp = current
        left = 0
        right = 0
        while temp+1 in s:
            s.remove(temp+1)
            right += 1
            temp += 1
        temp = current
        while temp-1 in s:
            s.remove(temp-1)
            left += 1
            temp -= 1

        output = max(output, left + right + 1)
    return output


# 110
def isBalanced(self, root):
    if not root: return True

    def helper(root):
        if not root: return 0
        l = helper(root.left)
        r = helper(root.right)
        if l == -1 or r == -1 or abs(l-r) > 1:
            return -1
        return max(l, r)+1

    return helper(root) >= 0

"""
Single hotel room booking
We have a single hotel room, and we want to help guests plan their stay by finding a check-in date that accommodate a desired length of stay.

We determine availability based on existing bookings, serialized as colon-separated pairs of integers.
The first integer is a check-in date, and the second is a check-out date
Each integer represents an offset since Jan 1, 2019.

E.g. '1:2' represents a booking where the check-in date is Jan 1st 2019, and the check-out date is Jan 2nd 2019.


Directions
Implement a method, booking_start_date(string bookings, int stay_length, int today)
that will return the first day that can accommodate a booking of length stay_length.

Examples:
Input: bookings: '1:3 4:6 8:15', stay_length: 1, today: 5
Output: 6
Input: bookings: '1:4 4:7 8:15', stay_length: 2, today: 5
Output: 15
Input: bookings: '1:2 6:7 8:15', stay_length: 1, today: 3
Output: 3
"""
# Single Hotel Room Booking
def HotelRoomBooking(t, stay_length, today):
    t = t.split(" ")
    t = [temp.split(":") for temp in t]
    for i in range (1, len(t)):
        if int(t[i][0])>today and int(t[i][0])-max(int(t[i-1][1]),today)>=stay_length:
            return max(int(t[i-1][1]), today)
    return max(int(t[-1][1]),today)
# print HotelRoomBooking('1:2 6:7 8:15', 1, 3)




# if only positive numbers
def shortestSubarraySumGreaterThanK(nums, target):
    if not nums: return 0
    output = float("inf")
    left, right = 0, 0
    totalSum = 0
    while right < len(nums):
        if totalSum <= target:
            totalSum += nums[right]
            right += 1
        else:
            output = min(output, right-left)
            totalSum -= nums[left]
            left += 1
    if output == float("inf"):
        return 0
    return output
#  print shortestSubarraySumGreaterThanK([2,1,3,4,5], 3)

#  if negativeNumer
def shortestSubarraySumGreaterThanKNeg(nums, target):
    if not nums: return 0
    dic = collections.defaultdict(int)
    totalSum = 0
    output = float("inf")
    for i in range (len(nums)):
        print i, output, totalSum
        totalSum += nums[i]
        if totalSum - target in dic:
            output = min(output, i-dic[totalSum-target]+1)
        dic[totalSum] = i
    if output == float("inf"):
        return 0
    return output
# print shortestSubarraySumGreaterThanKNeg([-3,2,1,3,4,5],3)


def longestSubarraySumGreaterThanK(nums, target):
    if not nums: return 0
    output = 0
    for i in range (len(nums)):
        totalSum = 0
        for j in range (i, len(nums)):
            totalSum += nums[j]
            if totalSum > target:
                output = max(output, j-i+1)
    return output
start = time.time()
print longestSubarraySumGreaterThanK([1,2,3,-1,-2,-3,4],3)
end = time.time()
print end-start

def longestSubarraySumGreaterThanK2(nums, target):
    if not nums: return 0
    output = 0
    for i in range (len(nums)):
        totalSum = sum(nums[i:])
        for j in range (len(nums)-1, i-1, -1):
            if totalSum > target:
                output = max(output, j-i+1)
            totalSum -= nums[j]
    return output

start = time.time()
print longestSubarraySumGreaterThanK2([1,2,3,-1,-2,-3,4],3)
end = time.time()
print end-start





def shortestSubarray(self, A, k):
    """
    :type A: List[int]
    :type K: int
    :rtype: int
    """
    if not A:
        return -1
    totalSum = [0]
    for n in A:
        totalSum.append(n+totalSum[-1])
    dq = collections.deque()
    output = float("inf")
    for i, v in enumerate (totalSum):
        while dq and totalSum[dq[-1]] >= v:
            dq.pop()
        while dq and v-totalSum[dq[0]] >= k:
            output = min(output, i-dq.popleft())
        dq.append(i)
    if output == float("inf"):
        return -1
    return output









# def compare(a, b):
#
# 	if a[0] == b[0]:
# 		return a[1] < b[1]
#
# 	return a[0] < b[0]

def findInd(preSum, n, val):
	l, h = 0, n - 1

	# To store required index value.
	ans = -1

	# If middle value is less than or equal
	# to val then index can lie in mid+1..n
	# else it lies in 0..mid-1.
	while l <= h:
		mid = (l + h) // 2
		if preSum[mid][0] <= val:
			ans = mid
			l = mid + 1

		else:
			h = mid - 1

	return ans

# Function to find largest subarray having
# sum greater than or equal to k.
def largestSub(arr, n, k):
	maxlen = 0
	preSum = []
	Sum = 0
	minInd = [None] * (n)
	for i in range(0, n):
		Sum = Sum + arr[i]
		preSum.append([Sum, i])
	preSum.sort(key = lambda x:[x[0], x[1]])

	# Update minInd array.
	minInd[0] = preSum[0][1]

	for i in range(1, n):
		minInd[i] = min(minInd[i - 1], preSum[i][1])
	print preSum
	print minInd
	Sum = 0
	for i in range(0, n):
		Sum = Sum + arr[i]

		if Sum > k:
			maxlen = i + 1

		# If sum is less than or equal to k,
		# then find if there is a prefix array
		# having sum that needs to be added to
		# current sum to make its value greater
		# than k. If yes, then compare length
		# of updated subarray with maximum
		# length found so far.
		else:
			ind = findInd(preSum, n, Sum - k - 1)
			print Sum-k-1, ind, arr[i]
			if ind != -1 and minInd[ind] < i:
				maxlen = max(maxlen, i - minInd[ind])

	return maxlen

# Driver code.
if __name__ == "__main__":

	arr = [-2, 1, 6, -3]
	n = len(arr)

	k = 5

	print(largestSub(arr, n, k))

# This code is contributed
# by Rituraj Jain
