"""
53.最大子数组和
给你一个整数数组 nums ,请你找出一个具有最大和的连续子数组（子数组最少包含一个元素）,返回其最大和。
子数组是数组中的一个连续部分。
输入:nums = [-2,1,-3,4,-1,2,1,-5,4]
输出:6
解释:连续子数组 [4,-1,2,1] 的和最大,为 6 。

"""
# 方法:动态规划,时间复杂度O(n),空间复杂度O(1)
# 定义dp数组,dp[i]表示以nums[i]结尾的最大子数组和,则有状态转移方程:dp[i] = max(dp[i-1] + nums[i], nums[i]),即要么将当前元素加入到之前的子数组中,要么重新开始一个新的子数组。
class Solution:
    def maxSubArray(self, nums):
        dp = [0] * len(nums)
        dp[0] = nums[0] # 初始化dp数组,dp[i]表示以nums[i]结尾的最大子数组和
        for i in range(1, len(nums)):
            dp[i] = max(dp[i-1] + nums[i], nums[i])
        return max(dp)


"""
56.合并区间
以数组 intervals 表示若干个区间的集合,其中单个区间为 intervals[i] = [starti, endi] 。
请你合并所有重叠的区间,并返回 一个不重叠的区间数组,该数组需恰好覆盖输入中的所有区间 。
输入:intervals = [[1,3],[2,6],[8,10],[15,18]]
输出:[[1,6],[8,10],[15,18]]
"""
# 方法:排序+遍历,时间复杂度O(nlogn),空间复杂度O(n)
# 首先对区间按照起始位置进行排序,然后遍历排序后的区间列表,如果当前区间的起始位置小于等于上一个区间的结束位置,
# 说明两个区间重叠,将它们合并为一个区间;否则,将当前区间添加到结果列表中。最后返回结果列表。

class Solution:
    def merge(self, intervals):
        intervals.sort(key=lambda x: x[0]) # 按照区间的起始位置进行排序
        merged = []
        for interval in intervals:
            # 如果结果列表为空或者当前区间的起始位置大于上一个区间的结束位置,说明当前区间与结果列表中的最后一个区间不重叠,将当前区间添加到结果列表中
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则,说明当前区间与结果列表中的最后一个区间重叠,将它们合并为一个区间,更新结果列表中的最后一个区间的结束位置为两个区间结束位置的较大值
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged


"""
189.轮转数组
给定一个整数数组 nums,将数组中的元素向右轮转 k 个位置,其中 k 是非负数。
输入:nums = [1,2,3,4,5,6,7], k = 3
输出:[5,6,7,1,2,3,4]
"""
# 方法:切片,时间复杂度O(n),空间复杂度O(n)

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        k = k % len(nums)
        nums[:] = nums[-k:] + nums[:-k] # 切片操作,将数组分为两部分,后一部分放在前面,前一部分放在后面


"""
41.缺失的第一个正数
给你一个未排序的整数数组 nums ,请你找出其中没有出现的最小的正整数。
请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案。
输入:nums = [1,2,0]
输出:3
输入:nums = [3,4,-1,1]
输出:2
"""
# 方法:哈希表,时间复杂度O(n),空间复杂度O(n)
class Solution:
    def firstMissingPositive(self, nums):
        num_set = set(nums) # 将数组转换为集合,方便查找
        for i in range(1, len(nums) + 1): # 从1开始遍历,找到第一个不在集合中的正整数
            if i not in num_set:
                return i
        return len(nums) + 1 # 如果1到n都在集合中,则返回n+1
    
    # 方法: 1.将每个正整数放到它应该在的位置上,即nums[i]应该放在nums[nums[i]-1]的位置上, 过滤与排原则,时间复杂度O(n),空间复杂度O(1)
    def firstMissingPositive(self, nums):
        # 方法:原地哈希,时间复杂度O(n),空间复杂度O(1)
        for i in range(len(nums)):
            # 将每个正整数放到它应该在的位置上,即nums[i]应该放在nums[nums[i]-1]的位置上, 过滤与排原则
            while nums[i] > 0 and nums[i] <= len(nums) and nums[nums[i] -1] != nums[i]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1] # 
        # 2.遍历数组, 找到第一个位置i上没有放置正确的正整数,即nums[i] != i+1,则返回i+1   
        for i in range(len(nums)):
            if nums[i] != i+1:
                return i+1
        return len(nums) + 1

"""
73.矩阵置零
给定一个 m x n 的矩阵,如果一个元素为 0 ,则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
输入:matrix = [[1,1,1],[1,0,1],[1,1,1]]
输出:[[1,0,1],[0,0,0],[1,0,1]]
"""
# 方法:哈希表,时间复杂度O(m*n),空间复杂度O(m+n)
# 时间复杂度O(m*n),空间复杂度O(m+n)
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        row = set()
        col = set()
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    row.add(i)
                    col.add(j)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i in row or j in col:
                    matrix[i][j] = 0
# 方法:使用第一行和第一列来记录需要置零的行和列,避免使用额外空间,时间复杂度O(m*n),空间复杂度O(1)
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m, n = len(matrix), len(matrix[0])
        first_row_has_zero = 0 in matrix[0]
        # 使用第一行和第一列来记录需要置零的行和列,避免使用额外空间
        for i in range(1, m):
            for j in range(n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        # 根据第一行和第一列的记录,将需要置零的行和列置零,注意要倒着遍历,避免提前把第一行或第一列改成0,误认为这一行或这一列要全部变成0
        for i in range(1, m):
            # 倒着遍历,避免提前把 matrix[i][0] 改成 0,误认为这一行要全部变成 0
            for j in range(n - 1, -1, -1):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        # 最后根据第一行的记录,将第一行置零
        if first_row_has_zero:
            for j in range(n):
                matrix[0][j] = 0

"""
54.螺旋矩阵
给你一个 m 行 n 列的矩阵 matrix ,请按照 顺时针螺旋顺序 ,返回矩阵中的所有元素。
输入:matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出:[1,2,3,6,9,8,7,4,5]
"""
# 时间复杂度O(m*n),空间复杂度O(m*n)
# 方法: 使用四个指针left, right, top, bottom来记录当前矩阵的边界,
# 每次按照顺时针的顺序遍历当前边界上的元素,然后更新边界,直到所有元素都被遍历完毕.
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        left, right, top, bottom = 0, n - 1, 0, m - 1
        res = []
        while left <= right and top <= bottom:
            # 从左到右
            for col in range(left, right + 1):
                res.append(matrix[top][col])
            top += 1
            # 从上到下
            for row in range(top, bottom + 1):
                res.append(matrix[row][right])
            right -= 1

            # 从右到左
            if top < bottom:
                for col in range(right, left-1, -1):
                    res.append(matrix[bottom][col])
                bottom -= 1
            # 从下到上
            if left < right:
                for row in range(bottom, top-1, -1):
                    res.append(matrix[row][left])
                left += 1
        return res     


"""
48.旋转图像
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。
你必须在 原地 旋转图像,这意味着你需要直接修改输入的二维矩阵。请不要 使用另一个矩阵来旋转图像。
输入:matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出:[[7,4,1],[8,5,2],[9,6,3]]
"""
# 方法:使用一个新的矩阵来存储旋转后的结果,然后将新矩阵的值复制回原矩阵,时间复杂度O(n^2),空间复杂度O(n^2)
class Solution:
    def rotate(self, matrix):
        n = len(matrix)
        # New[col][N-row-1] = Old[row][col]
        matrix_new = [[0]* n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix_new[j][n - i - 1] = matrix[i][j]
        matrix[:] = matrix_new
        return matrix

    # 方法:先转置矩阵,再反转每一行,时间复杂度O(n^2),空间复杂度O(1)
    def rotate(self, matrix):
        n = len(matrix)
        # 先转置矩阵,再反转每一行
        # 转置矩阵:New[col][row] = Old[row][col]
        for i in range(n):
            for j in range(i):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        # 反转每一行
        for row in matrix:
            row.reverse()
        return matrix


"""
240.搜索二维矩阵II
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
"""

# 方法:从矩阵的右上角开始寻找,如果当前元素等于目标值,则返回True;如果当前元素大于目标值,则说明目标值不可能在当前列,将列指针左移;
# 如果当前元素小于目标值,则说明目标值不可能在当前行,将行指针下移。重复这个过程,直到找到目标值或者超出矩阵边界。
# 时间复杂度O(m+n),空间复杂度O(1)
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        row, col = 0, n - 1 # 从右上角开始寻找
        while row < m  and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        return False





##################################
# 哈希
##################################

"""
1.两数之和
给定一个整数数组 nums 和一个目标值 target,请你在该数组中找出和为目标值的那 两个 整数,并返回他们的数组下标。
输入:nums = [2,7,11,15], target = 9
输出:[0,1]
"""
# 方法:哈希表,时间复杂度O(n),空间复杂度O(n)
class Solution:
    def twoSum(self, nums, target):
        index = {}
        for i, num in enumerate(nums):
            if target - num in index:
                return [index[target - num], i]
            index[num] = i
        return []


"""
49.字母异位词分组
给你一个字符串数组 strs ,将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
输入:strs = ["eat","tea","tan","ate","nat","bat"]
输出:[["bat"],["nat","tan"],["ate","eat","tea"]]
"""

# 方法:哈希表,时间复杂度O(n*klogk),空间复杂度O(n),其中n是字符串数组的长度,k是字符串的平均长度
# 对于每个字符串,我们将其排序后的结果作为哈希表的键,将原字符串添加到对应键的值列表中。最后返回哈希表中所有值列表的集合。
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        tmp = defaultdict(list)
        for char in strs:
            new_char = "".join(sorted(char))
            tmp[new_char].append(char)
        return list(tmp.values())

        
"""
i给定一个未排序的整数数组 nums ,找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
请你设计并实现时间复杂度为 O(n) 的算法解决此问题。
输入:nums = [100,4,200,1,3,2]
输出:4
"""


# 方法:哈希表,时间复杂度O(n),空间复杂度O(n)
# 1.首先将数组转换为集合,方便查找。
# 2.然后遍历集合中的每个数字,如果当前数字的前一个数字不在集合中,说明当前数字是一个连续序列的起点,我们就不断地寻找下一个连续的数字,直到找不到为止。
# 3.最后更新最长连续序列的长度。
class Solution:
    def longestConsecutive(self, nums):
        num_set = set(nums) # 将数组转换为集合,方便查找
        longest = 0
        for num in num_set:
            if num - 1 not in num_set: # 只有当num-1不在集合中时,才开始寻找以num为起点的连续序列
                length = 1
                while num + length in num_set: # 不断地寻找下一个连续的数字,直到找不到为止
                    length += 1
                longest = max(longest, length) # 更新最长连续序列的长度
        return longest
