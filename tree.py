
# 94. 二叉树的中序遍历
"""https://leetcode.cn/problems/binary-tree-inorder-traversal/description/
给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
示例 1:
输入:root = [1,null,2,3]
输出:[1,3,2]
示例 2:
输入:root = []
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# 构建测试树
#     1
#    / \
#   2   3
#  / \
# 4   5

# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)

# print(inorderTraversal(root))  # 输出: [4,2,5,1,3]

# 方法:递归,中序遍历:左根右
# (中序遍历顺序:左子树 → 根节点 → 右子树)
# 前序遍历:根左右
# 后序遍历:左右根
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        def dfs(root):
            if not root:
                return
            dfs(root.left) # 递归左子树
            res.append(root.val) # 访问根节点
            dfs(root.right) # 递归右子树
        dfs(root)
        return res
# dfs(1)->dfs(1.left)->dfs(1.left.left)->dfs(1.left.left.left) -> return
# res.append(4)
# dfs(1.left.left.right) -> return
# return to dfs(1.left)
# res.append(2)
# dfs(1.left.right) -> dfs(1.left.right.left) -> return
# res.append(5)
# dfs(1.left.right.right) -> return
# return to dfs(1)
# res.append(1)
# dfs(1.right) -> dfs(1.right.left) -> return
# res.append(3)
# return res = [4,2,5,1,3]

# 144. 二叉树的前序遍历
# 方法:递归,深度优先搜索，前序遍历:根左右
class Solution(object):
    def preorderTraversal(self, root):
        res = []
        def dfs(root):
            if not root:
                return
            res.append(root.val) # 访问根节点
            dfs(root.left) # 递归左子树
            dfs(root.right) # 递归右子树
        dfs(root)
        return res


"""
# 104. 二叉树的最大深度
https://leetcode.cn/problems/maximum-depth-of-binary-tree/description/
给定一个二叉树，找出其最大深度。
二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
说明: 叶子节点是指没有子节点的节点。
示例:
给定二叉树 [3,9,20,null,null,15,7]
    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 
"""

#方法:递归,深度优先搜索
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: # 递归终止条件:节点为空时,深度为0
            return 0
        else:
            l_height = self.maxDepth(root.left) # 递归左子树
            r_height = self.maxDepth(root.right) # 递归右子树
            return max(l_height,r_height) + 1 # 返回左右子树高度中的较大值+1
# 解释: 
# root=3, l =maxDepth(9)=1, r=maxDepth(20)
# max(1, maxDepth(20)) + 1
# root=20, l=maxDepth(15)=1, r=maxDepth(7)=1
# max(1,1)+1=2
# return max(1,2)+1=3
# 递归终止条件:节点为空时,深度为0
# 递归过程:
# 1. 从根节点开始,分别计算左右子树的高度
# 2. 返回左右子树高度中的较大值+1
# 时间复杂度:O(N),每个节点访问一次
# 空间复杂度:O(H),递归栈的空间,H为树的高度


"""# 226. 翻转二叉树
https://leetcode.cn/problems/invert-binary-tree/description/
翻转一棵二叉树。
输入:root = [4,2,7,1,3,6,9]
输出:[4,7,2,9,6,3,1]
"""

# 方法:递归,深度优先搜索
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        # 交换左右子树
        root.left, root.right = root.right, root.left
        # 递归交换左子树和右子树
        self.invertTree(root.left) # 递归左子树
        self.invertTree(root.right) # 递归右子树
        return root 


"""# 101. 对称二叉树
https://leetcode.cn/problems/symmetric-tree/description/
给定一个二叉树，检查它是否是镜像对称的。
例如，二叉树 [1,2,2,3,4,4,3] 是对称的。"""
# 方法:递归,深度优先搜索
# 1. 定义辅助函数isMirror(left, right),判断两个子树是否互为镜像
# 2. 递归比较左子树的左子节点和右子树的右子节点,以及左子树的右子节点和右子树的左子节点
# 3. 初始调用isMirror(root.left, root.right)
# 时间复杂度:O(N),每个节点访问一次
# 空间复杂度:O(H),递归栈的空间,H为树的高度
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def isMirror(left, right):
            if not left and not right: # 左右子树都为空
                return True
            elif not left or not right: # 只有一个子树为空
                return False
            # 比较左子树的左子节点和右子树的右子节点,以及左子树的右子节点和右子树的左子节点
            return left.val == right.val and isMirror(left.left, right.right) and isMirror(left.right, right.left)
        return isMirror(root.left, root.right)


# 543. 二叉树的直径
"""https://leetcode.cn/problems/diameter-of-binary-tree/description/
给定一棵二叉树，你需要计算它的直径长度。
一棵二叉树的直径长度是任意两个结点路径长度中的最大值。
这条路径可能穿过也可能不穿过根结点。
示例 : 
给定二叉树
          1
         / \
        2   3
       / \     
      4   5    
返回 3, 它的长度是路径 [4,2,1,3] 或者 [5,2,1,3]
注意:两结点之间的路径长度是以它们之间边的数目表示."""

# 方法:递归,深度优先搜索
# 1. 定义全局变量ans,初始化为1
# 2. 定义辅助函数depth(node),计算以node为根的子树的深度
# 3. 递归计算左子树的深度L和右子树的深度R,然后计算d_node即L+R+1 并更新ans
# 4. 返回该节点为根的子树的深度max(L, R) + 1
# 5. 初始调用depth(root)
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.ans = 1
        def depth(node):
            # 访问到空节点
            if not node:
                return 0
            # 左节点为根的子树的深度
            L = depth(node.left)
            # 有节点
            R = depth(node.right)
            # 计算d_node即L+R+1 并更新ans
            self.ans = max(self.ans, L+R+1)
            # 返回该节点为根的子树的深度
            return max(L, R) + 1
        
        depth(root)
        return self.ans - 1

# 102. 二叉树的层序遍历
"""https://leetcode.cn/problems/binary-tree-level-order-traversal/description/
给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
示例 1:
输入:root = [3,9,20,null,null,15,7]
输出:[[3],[9,20],[15,7]]
"""
# 方法:广度优先搜索, 通过队列实现，每次处理一层节点
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root: # 节点为空时,深度为0
            return [] 
        res = []
        queue = [root]
        while queue:
            length = len(queue)
            level = [] # 存储当前层级节点值
            for i in range(length): # 遍历当前层级节点
                node = queue.pop(0) # 弹出队列头部元素
                level.append(node.val) # 添加当前节点值
                if node.left:
                    queue.append(node.left) # 添加左孩子
                if node.right:
                    queue.append(node.right) # 添加右孩子

            res.append(level)
        return res


# 199. 二叉树的右视图
"""https://leetcode.cn/problems/binary-tree-right-side-view/description/
给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
示例 1:
输入:root = [1,2,3,null,5,null,4]
输出:[1,3,4]
"""
# 方法:广度优先搜索, 通过队列实现，每次处理一层节点,记录每层最后一个节点，同上述方法一样
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        res = []
        queue = [root]
        while queue:
            length = len(queue)
            for i in range(length):
                node = queue.pop(0)
                # 每层的最后一个节点
                if i == length - 1:
                    res.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return res


# 108. 将有序数组转换为二叉搜索树
"""https://leetcode.cn/problems/convert-sorted-array-to-binary-search-tree/description/
将一个按照升序排列的有序数组，转换为一棵高度平衡的二叉搜索树。
本题中，一个高度平衡二叉树是指一个二叉树每个节点 的左右两个子树的高度差的绝对值不超过 1。
示例:    
给定有序数组: [-10,-3,0,5,9],
一个可能的答案是:[0,-3,9,-10,null,5]
"""
# 方法:递归,深度优先搜索
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums: # 数组为空时
            return None
        mid = len(nums) // 2 # 选择中间元素作为根节点
        root = TreeNode(nums[mid]) # 创建根节点
        root.left = self.sortedArrayToBST(nums[:mid]) # 递归构建左子树
        root.right = self.sortedArrayToBST(nums[mid+1:]) # 递归构建右子树
        return root


# 98. 验证二叉搜索树
"""https://leetcode.cn/problems/validate-binary-search-tree/description/
给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
有效 二叉搜索树定义如下:
节点的左子树只包含 小于 当前节点的数。
节点的右子树只包含 大于 当前节点的数。 所有左子树和右子树自身必须也是二叉搜索树。
示例 1:
输入:root = [2,1,3]
输出:true
"""
# 方法:递归,深度优先搜索
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def helper(node, lower=float('-inf'),  upper=float('inf')):
            if not node:
                return True
            val = node.val # 节点值
            if val <= lower or val >= upper: # 节点值小于等于下界或大于等于上界
                return False
            # 右子树 > 根节点 > 左子树
            if not helper(node.right, val, upper): # 递归右子树,更新下界为当前节点值
                return False
            if not helper(node.left, lower, val): # 递归左子树,更新上界为当前节点值
                return False

            return True
        return helper(root)
    
# 230. 二叉搜索树中第K小的元素
"""https://leetcode.cn/problems/kth-smallest-element-in-a-bst/description/
给定一个二叉搜索树，编写一个函数 kthSmallest 来查找其中第 k 个最小的元素。
示例 1:   
输入: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
输出: 1
"""
# 方法:中序遍历，递归
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.count = 0
        self.result = None
        # 递归中序遍历
        def inorder(node):
            if not node or self.result is not None:
                return
            
            inorder(node.left)
            
            self.count += 1
            if self.count == k: # 找到第k个元素
                self.result = node.val
                return
            
            inorder(node.right)
        
        inorder(root)
        return self.result

# 114. 二叉树展开为链表
"""https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/description/
给你二叉树的根结点 root ，请你将它展开为一个单链表:
展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。
示例 1:
输入:root = [1,2,5,3,4,null,6]
输出:[1,null,2,null,3,null,4,null,5,null,6]
"""
# 方法:前序遍历, 递归
class Solution:
    def flatten(self, root: Optional[TreeNode]) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        res = []
        # 前序遍历, 递归
        def dfs(node):
            if not node:
                return
            res.append(node)  # 存放的是节点
            dfs(node.left)
            dfs(node.right)
        dfs(root)
        size = len(res)
        # 遍历节点, 构建链表
        for i in range(1, size):
            prev, cur = res[i-1], res[i]
            prev.left = None
            prev.right = cur

# 105. 从前序与中序遍历序列构造二叉树
"""https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/
根据一棵树的前序遍历与中序遍历构建二叉树。
示例 1:
输入:preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出:[3,9,20,null,null,15,7]
"""
# 方法:递归,深度优先搜索
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        # 前序遍历:根左右
        # 中序遍历:左根右
        if not preorder or not inorder:
            return None
        # 前序遍历的第一个元素是根节点,在中序遍历中找到根节点的位置,左边是左子树,右边是右子树
        def helper(pre, ino):
            if not pre:
                return None
            root = TreeNode(pre[0]) # 创建根节点
            indexs = ino.index(pre[0]) # 区分左右子树

            root.left = helper(pre[1:indexs+1], ino[:indexs]) # 递归构建左子树
            root.right = helper(pre[indexs+1:], ino[indexs+1:]) # 递归构建右子树
            return root # 返回根节点
        return helper(preorder, inorder)


# 437. 路径总和 III
"""https://leetcode.cn/problems/path-sum-iii/description/
给定一个二叉树的根节点 root ，和一个整数目标和 targetSum ，
求该二叉树中和为目标和的路径数。
路径 不需要从根节点开始，也不需要在叶子节点结束，
但路径方向必须是向下的（只能从父节点到子节点）。
示例 1:
输入:root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
输出:3
解释:和为 8 的路径有 3 条，如图所示。
"""
# 方法:递归,深度优先搜索
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        # 计算以当前节点为起点,和为targetSum的路径数
        def rootSum(root, targetSum):
            if root is None:
                return 0
            ret = 0 # 记录路径数
            if root.val == targetSum:
                ret += 1
            
            ret += rootSum(root.left, targetSum - root.val)
            ret += rootSum(root.right, targetSum - root.val)
            return ret
        # 如果root为空,直接返回0
        if root is  None:
            return 0
        ret = rootSum(root, targetSum) # 计算以当前节点为起点的路径数
        ret += self.pathSum(root.left, targetSum) # 递归计算左子树的路径数
        ret += self.pathSum(root.right, targetSum) # 递归计算右子树的路径数
        return ret
    
# 236. 二叉树的最近公共祖先
"""https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
最近公共祖先的定义为:“对于有根树 T 的两个结点 p 和 q,
最近的一个既是 p 和 q 的祖先又是 T 的祖先的节点。”
示例 1:
输入:root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
输出:3
解释:节点 5 和节点 1 都在二叉树中,且它们分别是不同的节点。
"""
# 方法:递归,深度优先搜索
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 如果当前节点为空,或者等于p或q,则返回当前节点
        if not root or root == p or root == q:
            return root
        # 递归左子树
        left = self.lowestCommonAncestor(root.left, p, q)
        # 递归右子树
        right = self.lowestCommonAncestor(root.right, p, q)
        # 如果左子树和右子树都不为空,则当前节点为最近公共祖先
        if left and right:
            return root
        # 否则返回非空的子树结果
        return left if left else right

# 124. 二叉树中的最大路径和
"""https://leetcode.cn/problems/binary-tree-maximum-path-sum/description/
路径 被定义为一条从树中任意节点出发，沿父节点-子节点连接，达到任意节点的序列。
同一个节点在一条路径序列中 至多出现一次 。
该路径 至少包含一个 节点，且不一定经过根节点。
路径和 是路径中各节点值的总和。
给你一个二叉树的根节点 root ，返回其 最大路径和 。
示例 1:
输入:root = [1,2,3]
输出:6
解释:最优路径是 2 -> 1 -> 3 ，路径和为 2 + 1 + 3 = 6
"""
# 方法:递归,深度优先搜索
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        self.max_sum = float("-inf") # 初始化最大路径和为负无穷
        def dfs(node):
            if not node:
                return 0
            left_gain = max(dfs(node.left), 0) # 左子树的最大贡献值
            right_gain = max(dfs(node.right), 0) # 右子树的最大贡献值
            # 节点的最大路径和取决于该节点的值与该节点的左右子节点的最大贡献值
            price_newpath = node.val + left_gain + right_gain
            self.max_sum = max(self.max_sum, price_newpath) # 更新最大路径和
            return node.val + max(left_gain, right_gain) # 向上传递的最大贡献值, 左右子节点的最大贡献值
        dfs(root)
        return self.max_sum
