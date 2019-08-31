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
