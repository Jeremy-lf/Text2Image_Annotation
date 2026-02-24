############
# 二分查找
###########


"""
35.搜索插入位置
给定一个排序数组和一个目标值,在数组中找到目标值,并返回其索引。如果目标值不存在于数组中,返回它将会被按顺序插入的位置。
输入: nums = [1,3,5,6], target = 5
输出: 2
"""

class Solution(object):
    def searchInsert(self, nums, target):
        # 二分查找
        low, high = 0, len(nums) - 1
        # 循环条件：low小于等于high
        while low <= high:
            mid = (low + high) // 2
            # 判断mid位置的值与target的关系
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return low



"""

240. 搜索二维矩阵
给你一个满足下述两条属性的 m x n 整数矩阵:

每行中的整数从左到右按非严格递增顺序排列。
每行的第一个整数大于前一行的最后一个整数。
给你一个整数 target ,如果 target 在矩阵中,返回 true ；否则,返回 false 
输入:matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
输出:true
"""

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m*n-1
        while left <= right:
            mid = (left + right) // 2
            mid_value = matrix[mid // n][mid % n] # 获取mid位置的值
            if mid_value == target:
                return True
            elif mid_value < target:
                left = mid + 1
            else:
                right = mid -1
        return False


"""
34. 在排序数组中查找元素的第一个和最后一个位置
给你一个按照非递减顺序排列的整数数组 nums,和一个目标值 target。请你找出给定目标值在数组中的开始位置和结束位置。
如果数组中不存在目标值 target, 返回 [-1, -1]。
你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。
"""

class Solution:
    def searchRange(self, nums, target):
        res = []
        flag = 0
        for i in range(len(nums)):
            if nums[i] == target:
                res.append(i)
                flag = 1
        if flag == 0:
            return [-1, -1]
        else:
            return [res[0], res[-1]]


# 1、首先,在 nums 数组中二分查找得到第一个大于等于 target的下标leftBorder；
# 2、在 nums 数组中二分查找得到第一个大于等于 target+1的下标, 减1则得到rightBorder；
# 3、如果开始位置在数组的右边或者不存在target,则返回[-1, -1] 。否则返回[leftBorder, rightBorder]
class Solution:
    def searchRange(self, nums: List[int], target: int):
        # 二分查找目标的左边界
        def binarySearch(nums, target):
            left, right = 0, len(nums)-1
            while left <= right:
                mid = (left+right) // 2
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return left  # 若存在target,则返回第一个等于target的值
        leftboard = binarySearch(nums, target) # 搜索左边界
        rightboard = binarySearch(nums, target+1) -1 # 搜索右边界
        # 如果左边界越界或者左边界的值不等于target,说明数组中不存在target,返回[-1, -1]
        if leftboard == len(nums) or nums[leftboard] != target:
            return [-1, -1]
        return [leftboard, rightboard]

"""

33.搜索旋转排序数组
整数数组 nums 按升序排列,数组中的值 互不相同. 在传递给函数之前,nums 在预先未知的某个下标 k向左旋转,例如, [0,1,2,4,5,6,7] 下标 3 上向左旋转后可能变为 [4,5,6,7,0,1,2] 。
请你在该数组中搜索 target,如果目标值存在,则返回它的索引,否则返回 -1
输入:nums = [4,5,6,7,0,1,2], target = 0
输出:4
你必须设计一个时间复杂度为 O(log n) 的算法解决此问题。

"""

class Solution:
    def search(self, nums, target):
        if not nums:
            return -1
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            # 判断中间元素是否在左边界内, 如果在左边界内,则说明左边界有序,否则右边界有序
            if nums[0] <= nums[mid]:
                # 判断目标值是否在左边界内,如果在左边界内,则继续在左边界内搜索,否则在右边界内搜索
                if nums[0] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # 右边有序
                if nums[mid] < target <= nums[-1]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1

"""
153. 寻找旋转排序数组中的最小值
已知一个长度为 n 的数组,预先按照升序排列,经由 1 到 n 次 旋转后,得到输入数组。例如,原数组 nums = [0,1,2,4,5,6,7] 经过 3 次旋转后变为 [4,5,6,7,0,1,2]。
请找出其中最小的元素。
输入: nums = [3,4,5,1,2]
输出: 1

时间复杂度:O(logn),空间复杂度:O(1)
"""
class Solution:
    def findMin(self, nums):
        # 方法:二分查找,通过比较中间元素和右边界元素的大小关系来判断最小值所在的区间,然后不断缩小范围,直到找到最小值。
        left, right = 0, len(nums)-1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid
        return nums[left]



#################################################################

                #   栈与队列
#################################################################


"""
20. 有效的括号
给定一个只包括 '(',')','{','}','[',']' 的字符串 s ,判断字符串是否有效。
输入,s = "(())"
输出,true
时间复杂度:O(n),空间复杂度:O(n)
"""
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {")": "(", "}": "{", "]": "["}
        # 遍历字符串中的每个字符,如果是右括号,则检查栈顶元素是否匹配；如果匹配,则弹出栈顶元素；如果是左括号,则按顺序添加到栈中。
        for char in s:
            if char in mapping:
                # 如果是右括号,则检查栈顶元素是否匹配；
                if not stack or stack[-1] != mapping[char]:
                    return False
                # 如果匹配,则弹出栈顶元素；
                stack.pop()
            else:
                # 如果是左括号,则按顺序添加到栈中；
                stack.append(char)
        return not stack

"""
# 32. 最长有效括号
给你一个只包含 '(' 和 ')' 的字符串,找出最长有效（格式正确且连续）括号子串的长度。
输入:s = "(()"
输出:2
"""

class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1]
        res = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                # 遇到右括号,弹出栈顶元素
                stack.pop()
                if not stack:
                    stack.append(i) # 栈为空,将当前索引入栈
                else:
                    res = max(res, i-stack[-1]) # 计算当前匹配的括号长度,更新最大值
        return res

"""
155. 最小栈
设计一个支持 push ,pop ,top 操作, 并能在常数时间内检索到最小元素的栈。
push(x) —— 将元素 x 推入栈中。 pop() —— 删除栈顶的元素。 top() —— 获取栈顶元素。 getMin() —— 检索栈中的最小元素。
输入:
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
输出:
[null,null,null,null,-3,null,0,-2]
"""
class Solution:
    def __init__(self):
        self.stack = []
        self.min_stack = [float('inf')]  # 辅助栈,存储当前最小值
    
    def push(self, x):
        self.stack.append(x)
        self.min_stack.append(min(x, self.min_stack[-1]))  # 更新辅助栈的最小值
    
    def pop(self):
        self.stack.pop()
        self.min_stack.pop()  # 同时弹出辅助栈的最小值
    
    def top(self):
        return self.stack[-1] # 返回栈顶元素

    def getMin(self):
        return self.min_stack[-1]

    
"""
394. 字符串解码
给定一个经过编码的字符串,返回它解码后的字符串.给定一个经过编码的字符串,返回它解码后的字符串。 
编码规则为: k[encoded_string],表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。
示例 1:
输入:s = "3[a]2[bc]"
输出:"aaabcbc"
"""
class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        current_num = 0
        current_str = ''

        for char in s:
            if char.isdigit():
                # 数字字符,更新当前数字
                current_num = current_num * 10 + int(char) # 处理多位数字字符串
            elif char == '[':
                # 左括号,将当前字符串和数字压入栈中
                stack.append((current_str, current_num))
                current_str = ''
                current_num = 0
            elif char == ']':
                # 右括号,弹出栈顶元素,并将当前字符串重复num次,更新当前字符串
                last_str, num = stack.pop()
                current_str = last_str + num * current_str
            else:
                # 普通字符,更新当前字符串
                current_str += char
        return current_str



"""
739. 每日温度
给定一个整数数组 temperatures ,表示每天的温度,返回一个数组 answer ,其中 answer[i] 是指对于第 i 天,要想观测到更高的温度
需要等待的天数,如果气温在这之后都不会升高,请在该位置用 0 来代替.
输入:temperatures = [73,74,75,71,69,72,76,73]
输出:[1,1,4,2,1,1,0,0]
"""
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        res = [0] * n
        stack = []  # 存储索引的单调栈

        for i in range(n):
            # 当当前温度大于栈顶索引对应的温度时,更新结果数组,并弹出栈顶索引
            while stack and temperatures[i] > temperatures[stack[-1]]:
                idx = stack.pop()
                # 计算天数差
                res[idx] = i - idx
            stack.append(i)

        return res


"""
84. 柱状图中最大的矩形
给定 n 个非负整数,用来表示柱状图中各个柱子的高度。每个柱子彼此相邻,且宽度为 1 。求在该柱状图中,能够勾勒出来的矩形的最大面积。
输入:heights = [2,1,5,6,2,3]
输出:10
"""
class Solution:
    def largestRectangleArea(self, heights):
        res = 0
        n = len(heights)
        for i in range(n):
            min_height = heights[i]
            # 从当前高度开始向后遍历
            for j in range(i, n):
                # 更新最小高度
                min_height = min(min_height, heights[j])
                # 计算面积
                area = min_height * (j-i+1)
                res = max(res, area)
        return res

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        # 方法:单调栈,在柱状图中,当当前柱子高度小于栈顶柱子高度时,说明当前柱子无法再向右扩展了,
        # 需要计算以栈顶柱子为最小高度的矩形面积,然后弹出栈顶柱子,继续比较当前柱子与新的栈顶柱子高度的关系,
        # 直到当前柱子高度大于等于栈顶柱子高度为止。
        stack = []
        heights = [0] + heights + [0]
        res = 0
        for i in range(len(heights)):
            # 当当前高度小于栈顶高度时,计算面积
            while stack and heights[stack[-1]] > heights[i]:
                # 计算以栈顶高度为最小高度的矩形面积
                tmp = stack.pop()
                res = max(res, (i-stack[-1]-1) * heights[tmp])
            stack.append(i)
        return res

#################################################################

                #   堆与优先队列
#################################################################

"""
215. 数组中的第 K 个最大元素
给你一个整数数组 nums 和一个整数 k,请你返回数组中第 k 个最大的元素。
输入:nums = [3,2,1,5,6,4], k = 2
输出:5

为什么使用小顶堆而不是大顶堆？因为大顶堆需要维护所有元素,而小顶堆只需要维护k个元素,节省空间。
另外,当堆的大小超过k时弹出堆顶,确保堆中始终是最大的k个元素,堆顶就是这k个中的最小值,即第k大。
时间复杂度:O(nlogn),空间复杂度:O(1)
"""
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        import heapq
        # # 使用堆排序找到第k大的元素
        # return heapq.nlargest(k, nums)[-1]
        # nums.sort(),return nums[-k]

        # 借助一个小顶堆来维护当前堆内元素的最小值,同时保证堆的大小为 k
        pq = []
        for num in nums:
            # 将当前元素推入堆中
            heapq.heappush(pq, num)
            # 如果堆的大小超过了k,则弹出堆顶元素
            if len(pq) > k:
                heapq.heappop(pq) # 弹出堆顶元素
        return pq[0] # 堆顶元素即为第k大的元素
    

"""
23. 前 K 个高频元素
给你一个整数数组 nums 和一个整数 k ,请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
时间复杂度:O(nlogk),空间复杂度:O(n)
"""

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        import heapq
        from collections import Counter
        # 统计每个元素的频率
        # count = Counter(nums)
        # 使用堆排序找到前k个高频元素
        # return heapq.nlargest(k, count.keys(), key=count.get)
        return [num for num, _ in Counter(nums).most_common(k)]
