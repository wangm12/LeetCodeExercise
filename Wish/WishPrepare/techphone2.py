# def MaxVlueUnconnectedSets(root):
#     if not root: return 0
#     def helper(root):
#         # return [include current Node, not include current Node]
#         if not root: return [0, 0]
#         left = helper(root.left)
#         right = helper(root.right)
#         returnLeft = left[1] + right[1] + root.val
#         returnRight = max(left[0], left[1]) + max(right[0], right[1])
#         return [returnLeft, returnRight]
#
#     return max(helper(root))
#
# def MaxLengthUnconnectedSets(root):
#     if not root: return 0
#     def helper(root):
#         # retrun [include current Node, not include current node]
#         if not root: return [0, 0]
#         left = helper(root.left)
#         right = helper(root.right)
#         returnLeft = left[1] + right[1] + 1
#         returnright = max(left[0], left[1]) + max(right[0], right[1])
#         return [returnLeft, returnRight]
#     retrun max(helper(root))
import collections
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

# Problem 0. Create a Tree
def CreateTreeInorderPostorder(inorder, postorder):
    if not inorder or not postorder or len(inorder) != len(postorder):
        return None
    def helper(inorder, postorder):
        if not inorder or not postorder:
            return None
        root = TreeNode(postorder.pop(-1))
        rootIndex = inorder.index(root.val)
        root.right = helper(inorder[rootIndex+1:], postorder)
        root.left = helper(inorder[:rootIndex], postorder)
        return root
    return helper(inorder, postorder)

def maxLength(root):
    if not root: return 0
    def helper(root):
        if not root: return [0, 0]
        left = helper(root.left)
        right = helper(root.right)
        returnLeft = left[0] + right[0] + 1
        returnRight = max(left) + max(right)
        return [returnRight, returnLeft]
    return max(helper(root))

inorder = [9,3,15,20,7]
preorder = [3,9,20,15,7]
postorder = [9,15,7,20,3]
root = CreateTreeInorderPostorder(inorder, postorder)
# print maxLength(root)

def splitString(s):
    if not s: return []
    count = collections.Counter(s)
    output = []
    occur = set()
    startIndex = 0
    for i in range (len(s)):
        if s[i] not in occur:
            occur.add(s[i])
        count[s[i]] -= 1
        if count[s[i]] == 0:
            occur.remove(s[i])
        if not occur:
            output.append(s[startIndex:i+1])
            startIndex = i+1
    return output
# print splitString("abcade")


# /*
#  * clockwise rotate
#  * first reverse up to down, then swap the symmetry
#  * 1 2 3     7 8 9     7 4 1
#  * 4 5 6  => 4 5 6  => 8 5 2
#  * 7 8 9     1 2 3     9 6 3
# */
# /*
#  * anticlockwise rotate
#  * first reverse left to right, then swap the symmetry
#  * 1 2 3     3 2 1     3 6 9
#  * 4 5 6  => 6 5 4  => 2 5 8
#  * 7 8 9     9 8 7     1 4 7
# */


def rightAdd(l1, l2):
    if not l1 or not l2: return []
    if not l1: return l2
    if not l2: return l1
    l1.reverse()
    l2.reverse()
    output = []
    while l1 or l2:
        temp = 0
        if l1: temp += l1.pop(0)
        if l2: temp += l2.pop(0)
        output += [int(i) for i in str(temp)]
    return output
# print rightAdd([1, 0, 9], [90, 788, 100, 5])

# def trafficLight(l):
#     if not l: return 0
#     # find first redLight
#     position = -1
#     for i in range (len(l)):
#         if l[i] == 1:
#             position = i
#             break
#     if position == -1: return [-1, -1]
#     output = [position, position]
#     count = 1
#     answer = 1
#     for i in range (position, len(l)):
#         if l[i] == 1:
#             count += 1
#             if answer < count:
#                 output = [position, i]
#                 answer = count
#         else:
#             count -= 1
#             if count <= 0:
#                 count = 0
#                 position = i + 1
#     return output
def trafficLight(l):
    if not l: return [-1, -1]
    position = -1
    for i in range (len(l)):
        if l[i] == 1:
            position = i
            break
    if position == -1: return [-1, -1]
    output = [position, position]
    maxLength = 1
    currentLength = 1
    for i in range (position + 1, len(l)):
        if l[i] == 1:
            currentLength += 1
            if maxLength < currentLength:
                output = [position, i]
                maxLength = currentLength
        else:
            currentLength -= 1
            if currentLength <= 0:
                currentLength = 0
                position = i + 1
    if len(l) - position > maxLength:
        return [position, len(l) - 1]
    else:
        return output
# print trafficLight([0,1,0,0,1,1])

def relative(r, name1, name2):
    if not r or not r[0] or len(r[0]) != 3:
        return []
    dicList = {}
    for relation in r:
        if relation[0] not in dicList:
            dicList[relation[0]] = {}
        if relation[2] not in dicList[relation[0]]:
            dicList[relation[0]][relation[2]] = []
        dicList[relation[0]][relation[2]].append(relation[1])
    visited = set()
    output = []
    currentRelation = [name1]
    def helper(name, relation):
        if (name, relation) in visited: return
        # if not currentRelation or currentRelation[-1] != name: currentRelation.append(name)
        if name == name2:
            output.append(" ".join(currentRelation))
        visited.add((name, relation))
        if name not in dicList: return
        for secondName in dicList[name]:
            for nameRelationSecondName in dicList[name][secondName]:
                currentRelation.append(nameRelationSecondName)
                currentRelation.append(secondName)
                helper(secondName, nameRelationSecondName)
                currentRelation.pop()
                currentRelation.pop()
    helper(name1, "self")
    return output


test = [
["L", "self", "L"],
["B", "brother", "L"],
["B", "son", "H"],
["M", "wife", "H"],
["L", "daugtor", "H"]
# ["H", "temp", "H"]
]
print relative(test, "B", "H")
