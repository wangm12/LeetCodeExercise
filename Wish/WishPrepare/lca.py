#lca
def lca(root, p, q):
    if not root: return None
    def helper(root):
        if not root: return None
        if root.val == p.val or root.val == q.val:
            return root
        left = helper(root.left)
        right = helper(root.right)
        if left and right: return root
        if not left: return right
        return left
    return helper(root)

def lcaNotInTree(root, p, q):
    if not root: return None
    meetp = []
    meetq = []
    def helper(root):
        if not root: return None
        temp = False
        if root.val == q.val:
            meetq.append(True)
            temp = True
        if root.val == p.val:
            meetp.append(True)
            temp = True
        left = helper(root.left)
        right = helper(root.right)
        if (left and right) or temp:
            return root
        if not left and right: return right
        if not right and left: return left
        if not temp and not left and not right: return None


    output = helper(root)
    if meetp and meetq: return output.val
    return None

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
inorder = [9,3,15,20,7]
preorder = [3,9,20,15,7]
postorder = [9,15,7,20,3]
root = CreateTreeInorderPostorder(inorder, postorder)
print lcaNotInTree(root, TreeNode(15), TreeNode(7))
