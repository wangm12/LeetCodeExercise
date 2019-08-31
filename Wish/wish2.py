import collections
def ranking(n):
    if not n: return 0
    def f(n):
        output = []
        for i in range (1, n+1):
            if not output:
                output.append(i)
                continue
            output.append(i*output[-1])
        return output
    fact = f(n)
    dp = [[0 for i in range (n)] for i in range (n)]
    for place in range (n):
        for people in range (place, n):
            if place == 0: dp[place][people] = 1
            elif people == place: dp[place][people] = fact[place]
            else:
                dp[place][people] = (place+1)*dp[place][people-1] + (place+1)*dp[place-1][people-1]
    return sum([x[n-1] for x in dp])

# print ranking(2)
# ab,c;ac,b;bc,a;a,bc;b,ac;c,ab;

def generate(s):
    if not s: return []
    l = [c for c in s]
    output = []
    current = []
    def helper(index):
        if current:
            output.append("".join(current[:]))
        for i in range (index, len(l)):
            current.append(l[i])
            helper(i+1)
            current.pop()
    helper(0)
    return output
# print generate("abca")


def addNumberRight(num1, num2):
    output = []
    while num1 or num2:
        temp = 0
        if num1: temp += num1.pop()
        if num2: temp += num2.pop()
        while temp >= 10:
            output.append(temp%10)
            temp = temp//10
        output.append(temp)
    return output[::-1]
# print addNumberRight([1,0,9], [90,788,100,5])

def TrafficLight(arr):
    if not arr: return [-1, -1]
    position = -1
    for i, light in enumerate(arr):
        if light == "r":
            position = i
            break
    if position == -1: return [-1, -1]
    output = [position, position]
    currentMax = 1
    maxLength = 1
    for i in range (position+1, len(arr)):
        if arr[i] == "r":
            currentMax += 1
            if currentMax > maxLength:
                maxLength = currentMax
                output = [position, i]
        elif arr[i] == "g":
            currentMax -= 1
            if currentMax == 0:
                currentMax += 1
                position = i
    if len(arr)-position > maxLength:
        return [position, len(arr)-1]
    return output
# print TrafficLight(["l","l","r","l","r"])


def findBot(log, t, a):
    if not log or not log[0] or len(log[0]) != 2: return []
    bot = set()
    count = collections.defaultdict(int)
    dq = collections.deque()
    startTime = 0
    for id, time in log:
        while time - startTime > t:
            rt, rid = dq.popleft()
            count[rid] -= 1
            if not count[rid]: del(count[rid])
            startTime = rt
        dq.append((time, id))
        count[id] += 1
        if count[id] >= a:
            bot.add(id)
    return list[bots]


def LSLIS(arr):
    if not arr: return 0
    lis = [1 for i in range (len(arr))]
    ls = [i for i in arr]
    for i in range (len(arr)):
        for j in range (0, i):
            if arr[i] > arr[j]:
                lis[i] = max(lis[i], 1+lis[j])
                ls[i] = max(ls[i], arr[i]+ls[j])
    ml = max(lis)
    output = -float("inf")
    for i in range (len(lis)):
        if lis[i] == ml:
            output = max(output, ls[i])
    return output
print LSLIS([10,9,2,5,3,7,101,18])
