```python

import os
import sys
from collections import defaultdict


######### 滑动窗口  #############################################

######### 滑动窗口  #############################################

"""
3. 无重复字符的最长子串
给定一个字符串 s ,请你找出其中不含有重复字符的 最长 子串 的长度。

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc",所以其长度为 3。

# 双指针滑动窗口
在每一步的操作中,我们会将左指针向右移动一格,表示 我们开始枚举下一个字符作为起始位置,然后我们可以不断地向右移动右指针,
但需要保证这两个指针对应的子串中没有重复的字符。在移动结束后,这个子串就对应着 以左指针开始的,不包含重复字符的最长子串。
我们记录下这个子串的长度；

"""

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = left = 0

        window = set()  # 维护从下标 left 到下标 right 的字符
        for right, c in enumerate(s):
            while c in window:  # 加入 c 会导致窗口内有重复元素
                window.remove(s[left])  
                left += 1  # 缩小窗口
            window.add(c)
            ans = max(ans, right-left+1) # 更新窗口长度最大值
            
        return ans

if __name__ == "__main__":
    s = "bbbb"
    solution = Solution()
    result = solution.lengthOfLongestSubstring(s)
    print(result)  # 输出: 3



"""
438. 找到字符串中所有字母异位词
输入: s = "cbaebabacd", p = "abc"
输出: [0,6]
解释:
起始索引等于 0 的子串是 "cba", 它是 "abc" 的异位词。
起始索引等于 6 的子串是 "bac", 它是 "abc" 的异位词。

滑动窗口:通过维护一个固定长度的窗口（长度为 p_len）,在 s 上滑动,避免每次重新统计整个窗口的字符频率。
字符频率统计:用长度为 26 的数组表示 26 个小写字母的频率,通过比较两个数组是否相等来判断是否为异位词。
时间复杂度:O(n),其中 n 是 s 的长度。初始化统计频率为 O(m)（m 是 p 的长度）,滑动窗口遍历为 O(n - m),总体为 O(n)。
空间复杂度:O(1),使用了固定大小的数组（26 个元素）。

"""

class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_len, s_len = len(p), len(s)
        if p_len > s_len:
            return []

        ans = []
        p_count = [0] * 26
        s_count = [0] * 26
        for i in range(p_len):
            p_count[ord(p[i]) - ord('a')] += 1
            s_count[ord(s[i]) - ord('a')] += 1
        if p_count == s_count:
            ans.append(0)

        for i in range(s_len - p_len):
            s_count[ord(s[i]) - 97] -= 1
            s_count[ord(s[i + p_len]) - 97] += 1
            
            if s_count == p_count:
                ans.append(i + 1)
        return ans
    








######### 哈希表  #############################################
#                       #
######### 哈希表  #############################################

"""
1. 两数之和
给定一个整数数组 nums 和一个整数目标值 target,请你在该数组中找出 和为目标值 target  的那 两个 整数,并返回它们的数组下标。
输入:nums = [2,7,11,15], target = 9
输出:[0,1]
解释:因为 nums[0] + nums[1] == 9 ,返回 [0, 1] 。
"""

# 暴力解法
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] + nums[j] == target:
                    return [i, j]
        
        return []


# 方法二:哈希表
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        indexs = dict()
        for i, num in enumerate(nums):
            if target - num in indexs:
                return [i, indexs[target - num]]
            indexs[num] = i
        return []

"""
49. 字母异位词分组
给你一个字符串数组,请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。

输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]

输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
"""


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dic = dict()
        for s in strs:
            new_s = ''.join(sorted(s))
            if new_s in dic:
                dic[new_s].append(s)
            else:
                dic[new_s] = []
                dic[new_s].append(s)
        return list(dic.values())
        


"""
128. 最长连续序列

给定一个未排序的整数数组 nums ,找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

输入:nums = [100,4,200,1,3,2]
输出:4
解释:最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。

方法思路
使用哈希集合存储所有数字:首先将所有数字存入一个集合中,这样可以在 O(1) 时间内检查某个数字是否存在。
遍历每个数字:对于每个数字,检查它是否是某个连续序列的起点。即,如果当前数字的前一个数字（num - 1）不在集合中,那么当前数字可能是一个序列的起点。
扩展序列:如果当前数字是序列的起点,则向后查找连续的数字（num + 1, num + 2, ...）,直到序列不再连续为止,记录下序列的长度和序列本身。
更新最长序列:在遍历过程中,维护一个最长序列的变量,每次找到更长的序列时更新它。
"""

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
            # 找到新起点
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1
                 # 扩展序列
                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1
                # 更新最长序列
                longest_streak = max(longest_streak, current_streak)










###############################################################
#
#                           双指针   
#                       
###############################################################

"""
283. 移动零
给定一个数组 nums,编写一个函数将所有 0 移动到数组的末尾,同时保持非零元素的相对顺序。

输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
"""
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        left = right = 0 # 双指针,起始一样
        while right < n:
            if nums[right] != 0:
                nums[left], nums[right] = nums[right], nums[left] # 将左指针的零与右指针的非零数交换
                left += 1
            right += 1
        return nums
        
"""
11. 盛最多水的容器
给定一个长度为 n 的整数数组 height 。有 n 条垂线,第 i 条线的两个端点是 (i, 0) 和 (i, height[i]) 。

找出其中的两条线,使得它们与 x 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

双指针
"""

class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r = 0, len(height) - 1 # 左右边界
        ans = 0
        while l < r:
            area = min(height[l], height[r]) * (r-l) # 计算面积
            ans = max(ans, area) # 记录最大面积

            if height[l] <= height[r]:
                l += 1 # 如果左边<右边,则右移
            else:
                r -= 1
        return ans
        

"""
15. 三数之和

给你一个整数数组 nums ,判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ,同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。

注意:答案中不可以包含重复的三元组。

输入:nums = [-1,0,1,2,-1,-4]
输出:[[-1,-1,2],[-1,0,1]]
解释:
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0 。
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0 。
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0 。
不同的三元组是 [-1,0,1] 和 [-1,-1,2] 。
注意,输出的顺序和三元组的顺序并不重要。


思路:双指针问题, 先排序, 然后双指针, 注意去重
"""

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if(nums == None or len(nums) < 3):
            return []
        nums.sort()
        res = []
        for i in range(len(nums)):
            if(nums[i] > 0):
                return res
            # 去重第一部分
            if(i > 0 and nums[i] == nums[i-1]):
                continue
            left, right = i + 1, len(nums) - 1
            while(left < right):
                # 因为数组已经排序, 当nums[i]确定后, 整体大于0, 那么只能减小right对应的值
                if(nums[i] + nums[left] + nums[right] > 0):
                    right = right - 1
                # 因为数组已经排序, 当nums[i]确定后, 整体小于0, 那么只能增大left对应的值
                elif(nums[i] + nums[left] + nums[right] < 0):
                    left = left + 1
                else:
                    res.append([nums[i], nums[left], nums[right]])
                    # 高效去重第二部分, 一般情况下找到三数之和为0后, 应该左指针右移一位同时右指针左移一位, 然后继续搜索
                    # 但是题目返回数组要求不能重复, 左指针右移一位后可能值没变, 所以要确定值变为止, 右指针同理
                    while(left < right and nums[left + 1] == nums[left]):
                        left = left + 1
                    while(left < right and nums[right - 1] == nums[right]):
                        right = right - 1
                    left = left + 1
                    right = right - 1
        return res



"""
42. 接雨水
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图,计算按此排列的柱子,下雨之后能接多少雨水。
输入:height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出:6
解释:上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图,在这种情况下,可以接 6 个单位的雨水（蓝色部分表示雨水）。 

当两个指针没有相遇时,进行如下操作:
使用 height[left] 和 height[right] 的值更新 leftMax 和 rightMax 的值；
如果 height[left]<height[right],则必有 leftMax<rightMax,下标 left 处能接的雨水量等于 leftMax−height[left],将下标 left 处能接的雨水量加到能接的雨水总量,然后将 left 加 1（即向右移动一位）；
如果 height[left]≥height[right],则必有 leftMax≥rightMax,下标 right 处能接的雨水量等于 rightMax−height[right],将下标 right 处能接的雨水量加到能接的雨水总量,然后将 right 减 1（即向左移动一位）。
"""

class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        left, right = 0, len(height) - 1
        leftMax = rightMax = 0

        while left < right:
            # 如果 height[left] < height[right],说明左边的最大高度 leftMax 决定了当前位置的储水量。计算 leftMax - height[left] 并加到 ans 中,然后 left 右移。
            leftMax = max(leftMax, height[left])
            rightMax = max(rightMax, height[right])
            # 如果 height[left] < height[right],则 leftMax 是 left 位置的约束（因为右边有更高的柱子 height[right] 挡着,水的高度不会超过 leftMax）。
            if height[left] < height[right]:
                ans += leftMax - height[left]
                left += 1
            else:
                ans += rightMax - height[right]
                right -= 1
        return ans
    



###############################################################
#
#                           字串   
#                       
###############################################################

"""
560. 和为 K 的子数组

给你一个整数数组 nums 和一个整数 k ,请你统计并返回 该数组中和为 k 的子数组的个数 。
子数组是数组中元素的连续非空序列。

输入:nums = [1,1,1], k = 2
输出:2
"""
# 本题是连续子数组和问题,可以用前缀和转化。
# 滑动窗口需要满足单调性,当右端点元素进入窗口时,窗口元素和是不能减少的。
# 时间复杂度:O(n),其中 n 为 nums 的长度。
# 空间复杂度:O(n)。


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        # 构造前缀和
        s = [0] * (len(nums) + 1)
        for i, x in enumerate(nums):
            s[i+1] = s[i] + x
        
        ans = 0
        cnt = defaultdict(int) # 记录每个元素出现的次数,默认是0
        for sj in s:
            ans += cnt[sj - k] # 累计cnt中不为零的数量,k+j=sj转化为sj-k
            cnt[sj] += 1
        return ans
    


"""
76. 最小覆盖子串
给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串,则返回空字符串 "" 。
输入:s = "ADOBECODEBANC", t = "ABC"
输出:"BANC"


初始化阶段:
need字典统计t中每个字符的出现次数
window字典记录当前窗口中t字符的计数,初始为0
valid计数器记录当前窗口中满足t要求的字符种类数

滑动窗口流程:
右指针移动:扩大窗口,统计窗口内t字符的出现次数
有效字符判断:当某个字符在窗口中的数量达到t中的要求时,valid增加
窗口收缩:当valid == len(need)时,说明当前窗口已覆盖t
    记录当前有效窗口的位置和长度
    左指针右移,尝试缩小窗口
    移出字符时更新窗口计数和有效计数器

结果返回:
如果找到有效窗口,返回最小子串
否则返回空字符串
"""

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # 特殊情况处理
        if not s or not t or len(s) < len(t):
            return ""

        # 统计t中字符的出现次数
        need = Counter(t)
        # 初始化窗口内字符计数,只包含t中的字符
        window = {char: 0 for char in need}
        left, valid = 0, 0  # 窗口左指针和有效字符计数器
        min_len = float('inf')  # 最小子串长度
        start = 0  # 最小子串起始位置

        # 遍历字符串s的每个字符作为窗口右指针
        for right in range(len(s)):
            char = s[right]
            # 只处理t中存在的字符
            if char in need:
                window[char] += 1
                # 当该字符数量满足t中的要求时,有效计数器+1
                if window[char] == need[char]:
                    valid += 1
            
            # 当所有t中的字符都满足数量要求时,尝试收缩窗口
            while valid == len(need):
                # 更新最小子串信息
                if right - left + 1 < min_len:
                    min_len = right - left + 1
                    start = left
                
                # 移出窗口左端字符
                left_char = s[left]
                if left_char in need:
                    # 如果移出后该字符数量不满足要求,有效计数器-1
                    if window[left_char] == need[left_char]:
                        valid -= 1
                    window[left_char] -= 1
                left += 1

        return s[start:start+min_len] if min_len != float('inf') else ""



###############################################################
#
#                           数组   
#                       
###############################################################

"""
53. 最大子数组和
给你一个整数数组 nums ,请你找出一个具有最大和的连续子数组（子数组最少包含一个元素）,返回其最大和。

子数组是数组中的一个连续部分。

输入:nums = [-2,1,-3,4,-1,2,1,-5,4]
输出:6
解释:连续子数组 [4,-1,2,1] 的和最大,为 6 。

思路:
动态规划
dp[i] = dp[i-1] + nums[i] if dp[i-1] > 0 else nums[i]
"""
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        total = 0
        maxnum = -float('inf') # 初始化一个负无穷
        for i in range(len(nums)):
            total += nums[i]
            if total > maxnum:  # 更新累计最大和
                maxnum = total
            
            if total <= 0: # 如果total<=0,则重置最大子序列的起始位置
                total = 0
        return maxnum
    
class Solution(object):
    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(1,len(nums)):
            nums[i] = nums[i] + max(nums[i-1],0)
        return max(nums)
    

class Solution(object):
    def maxSubArray(self, num):
        dp = [0]*len(num)
        dp[0] = num[0]

        for i in range(1,len(num)):
            dp[i] = max(num[i],dp[i-1]+num[i])
            # print(dp[i-1]+dp[i])
        
        return max(dp)
    


"""
56. 合并区间

以数组 intervals 表示若干个区间的集合,其中单个区间为 intervals[i] = [starti, endi] 。
请你合并所有重叠的区间,并返回 一个不重叠的区间数组,该数组需恰好覆盖输入中的所有区间 
输入:intervals = [[1,3],[2,6],[8,10],[15,18]]
输出:[[1,6],[8,10],[15,18]]
解释:区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].


时间复杂度:O(nlogn),其中 n 为区间的数量。排序的时间复杂度为 O(nlogn),合并区间的时间复杂度为 O(n)。   
"""

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda x: x[0])  # 排序

        merged = []
        for interval in intervals:
            # 如果列表为空,或者当前区间与上一区间不重合,直接添加
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则的话,我们就可以与上一区间进行合并
                merged[-1][1] = max(merged[-1][1], interval[1])

        return merged


"""
189. 轮转数组
给定一个整数数组 nums,将数组中的元素向右轮转 k 个位置,其中 k 是非负数。

示例 1:

输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]

"""

class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # method 1
        # length_num = len(nums)
        # nums[:] = nums[-(k%length_num):] + nums[:-(k%length_num)]

        def reverse(i, j):
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        
        n = len(nums)
        k %= n ## 轮转 k 次等于轮转 k % n 次
        reverse(0, n-1)
        reverse(0, k-1)
        reverse(k, n-1)


"""
41. 缺失的第一个正数

给你一个未排序的整数数组 nums ,请你找出其中没有出现的最小的正整数。

请你实现时间复杂度为 O(n) 并且只使用常数级别额外空间的解决方案
输入:nums = [1,2,0]
输出:3
解释:范围 [1,2] 中的数字都在数组中。
"""

class Solution(object):
    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        遍历一次数组把大于等于1的和小于数组大小的值放到原数组对应位置,然后再遍历一次
        数组查当前下标是否和值对应,如果不对应那这个下标就是答案,否则遍历完都没出现那
        么答案就是数组长度加1。
        # nums[x-1] = x [1,2,3,...] i
        '''
      
        for i in range(len(nums)):
            while nums[i] > 0 and  nums[i] <= len(nums) and nums[nums[i] - 1] != nums[i]:
                nums[nums[i]-1],nums[i] = nums[i],nums[nums[i]-1]
        for i in range(len(nums)):
            if nums[i] != i + 1:
                return i+1
        return len(nums) + 1
    

"""
73. 矩阵置零
给定一个 m x n 的矩阵,如果一个元素为 0 ,则将其所在行和列的所有元素都设为 0 。请使用 原地 算法。
"""
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        col = set()
        row = set()
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    row.add(i)
                    col.add(j)
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i in row or j in col:
                    matrix[i][j] = 0



"""
54. 螺旋矩阵
方法二:按层模拟
可以将矩阵看成若干层,首先输出最外层的元素,其次输出次外层的元素,直到输出最内层的元素。
"""

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        if not matrix or not matrix[0]:
            return list()
        
        rows, cols = len(matrix), len(matrix[0])
        order = list()
        left, right, top, bottom = 0, cols-1, 0, rows-1
        while left <= right and top <= bottom:
            # 左到右
            for col in range(left, right+1):
                order.append(matrix[top][col])
            # 上到下
            for row in  range(top +1, bottom+1):
                order.append(matrix[row][right])
            
            if left < right and top < bottom:
                for col in range(right-1, left, -1):
                    order.append(matrix[bottom][col])
                
                for row in range(bottom, top, -1):
                    order.append(matrix[row][left])
            left,right,top, bottom = left+1, right-1, top+1, bottom-1
        return order
    
"""

48. 旋转图像
给定一个 n × n 的二维矩阵 matrix 表示一个图像。请你将图像顺时针旋转 90 度。

"""
 
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        # 辅助数组,旋转后位置对应New[col][n-row-1]=N[row][col]
        matrix_new = [[0]* n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix_new[j][n-i-1] = matrix[i][j] # 旋转90度后的新位置
        matrix[:] = matrix_new


"""
240. 搜索二维矩阵 II

编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target 。该矩阵具有以下特性:

每行的元素从左到右升序排列。
每列的元素从上到下升序排列
"""
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 方法一,循环遍历
        # for row in matrix:
        #     for ele in row:
        #         if ele == target:
        #             return True
        # return False
        # 时间复杂度:O(mn)。空间复杂度:O(1)。

        # 二分查找
        for row in matrix:
            idx = bisect.bisect_left(row, target)
            if idx < len(row) and row[idx] == target:
                return True
        return False
        # 时间复杂度:O(mlogn)。


################################

# 回溯算法
# 回溯法:一种通过探索所有可能的候选解来找出所有的解的算法。如果候选解被确认不是一个解（或者至少不是最后一个解）,回溯算法会通过在上一步进行一些变化抛弃该解,即回溯并且再次尝试。

# 找到一个可能存在的正确的答案；
# 在尝试了所有可能的分步方法后宣告该问题没有答案。
################################

"""
46.全排列
给定一个不含重复数字的数组 nums ,返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

输入:nums = [1,2,3]
输出:[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

"""

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtree(nums, tmp):
            # 遍历完nums中的元素,然后添加全排列到res
            if not nums:
                res.append(tmp)
                return 
            # nums中每个元素都作为第一个元素,然后递归调用
            for i in range(len(nums)):
                backtree(nums[:i] + nums[i+1:], tmp + [nums[i]]) # 遍历所有选择
        backtree(nums, [])
        return res
    
"""
78.子集
给你一个整数数组 nums ,数组中的元素 互不相同 。返回该数组所有可能的子集（幂集）。

解集 不能 包含重复的子集。你可以按 任意顺序 返回解集。
"""

class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # res = []
        # def backtrack(nums,i,tmp):
        #     res.append(tmp)
        #     for j in range(i,len(nums)):
                  # 注意这里的j+1,因为我们已经选择了nums[j],所以下一次应该从j+1开始选择
        #         backtrack(nums,j+1,tmp+[nums[j]])
        # backtrack(nums,0,[])
        # return res



        # import itertools
        # res = []
        # for i in range(len(nums)+1):
        #     for tmp in itertools.combinations(nums,i):
        #         res.append(tmp)
        # return res

        res = [[]]
        
        for i  in nums:
            res = res + [[i] + num for num in res]
        return res
    


"""
17. 电话号码的字母组合
给定一个仅包含数字 2-9 的字符串,返回所有它能表示的字母组合。答案可以按 任意顺序 返回。

给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。
输入:digits = "23"
输出:["ad","ae","af","bd","be","bf","cd","ce","cf"]


方法一:回溯
当题目中出现 “所有组合” 等类似字眼时,我们第一感觉就要想到用回溯。

定义函数 backtrack(combination, nextdigit),当 nextdigit 非空时,对于 nextdigit[0] 中的每一个字母 letter,
执行回溯 backtrack(combination + letter,nextdigit[1:],直至 nextdigit 为空。最后将 combination 加入到结果中


"""

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        phone = {'2':['a','b','c'],
                 '3':['d','e','f'],
                 '4':['g','h','i'],
                 '5':['j','k','l'],
                 '6':['m','n','o'],
                 '7':['p','q','r','s'],
                 '8':['t','u','v'],
                 '9':['w','x','y','z']}
        res = []
        def backtrack(nextdigit, conbination):
            # 终止条件
            if len(nextdigit) == 0:
                res.append(conbination)
            else:
                for letter in phone[nextdigit[0]]:
                    backtrack(nextdigit[1:], conbination + letter)
        backtrack(digits, '')
        return res
    

"""
39.组合总和
给你一个 无重复元素 的整数数组 candidates 和一个目标整数 target ,找出 candidates 中可以使数字和为目标数 target 的 所有 不同组合 ,并以列表形式返回。你可以按 任意顺序 返回这些组合。

"""
class Solution(object):
    def combinationSum(self, candidates, target):
        res = []
        def backtree(candidates, tmp):
            if sum(tmp) == target:
                res.append(tmp)
                return
            if sum(tmp) > target:
                return
            # 遍历所有节点,选择当前节点作为路径的一部分
            for i in range(len(candidates)):
                # 选择当前节点candidates[i],然后继续往下走
                backtree(candidates[i:], tmp + [candidates[i]])
        backtree(candidates, [])
        return res
        
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        def backtrack(candidates,tmp):
            # 找到答案
            if sum(tmp) == target:
                res.append(tmp[:])
            # 路径寻找的终止条件
            if sum(tmp) > target:
                return
            for i in range(len(candidates)):
                tmp.append(candidates[i]) # 注意这里的i,我们经选择candidates[i],可重复采样,直至sum(tmp)>target,然后执行remove(candidates[i]),循环下一个元素
                backtrack(candidates[i:],tmp)
                tmp.remove(candidates[i])
                print(tmp)
        backtrack(candidates,[])
        return res

"""
22.括号生成
数字 n 代表生成括号的对数,请你设计一个函数,用于能够生成所有可能的并且 有效的 括号组合。
输入:n = 3
输出:["((()))","(()())","(())()","()(())","()()()"]
"""

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        def backtree(tmp, left, right):
            # 终止条件
            if left == 0 and right == 0:
                res.append(tmp)
                return
            # 如果左括号还有剩余,就可以添加左括号
            if left > 0:
                backtree(tmp + "(", left - 1, right)
            # 如果右括号剩余数量大于左括号剩余数量,就可以添加右括号
            if right > left:
                backtree(tmp + ")", left, right - 1)
        backtree("", n, n)
        return res

"""
79. 单词搜索
给定一个 m x n 二维字符网格 board 和一个字符串单词 word 。如果 word 存在于网格中,返回 true ；否则,返回 false 。

单词必须按照字母顺序,通过相邻的单元格内的字母构成,其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。
输入:board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']], word = "ABCCED"
输出:true
"""

class Solution(object):    
    def exist(self, board: List[List[str]], word: str) -> bool:
        def dfs(i, j, word, visit):
            # 单词是否出现在以i,j为起点的网格中
            if len(word) == 1:
                return word[0] == board[i][j]
            elif board[i][j] != word[0]:
                return False
            visit[i][j] = True
            for (x,y) in [(i-1,j),(i+1,j),(i,j+1),(i,j-1)]: # 对四个方向进行搜索
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and not visit[x][y]:
                    if dfs(x, y, word[1:], visit):
                        return True
            visit[i][j] = False

        # direction = [(0,1), (0, -1), (1, 0), (-1, 0)]
        visit = [[False]*len(board[0]) for _ in range(len(board))]
        for i in range(len(board)): # 遍历所有格子作为单词起点
            for j in range(len(board[0])):
                
                if dfs(i,j,word,visit): 
                    return True
        return False
    


"""
131. 分割回文串

给你一个字符串 s,请你将 s 分割成一些 子串,使每个子串都是 回文串 。返回 s 所有可能的分割方案。
输入:s = "aab"
输出:[["a","a","b"],["aa","b"]]
"""
class Solution:
    def partition(self, s: str) -> List[List[str]]: 
        res = []
        def is_palindrome(subs):
            return subs == subs[::-1]
        
        def backtree(s, tmp):
            # 终止条件
            if not s:
                res.append(tmp)
                return
            # 遍历所有节点,选择当前节点作为路径的一部分
            for i in range(1, len(s)+1):
                # 选择当前节点s[:i],然后继续往下走
                if is_palindrome(s[:i]):
                    backtree(s[i:], tmp + [s[:i]])
        backtree(s, [])
        return res


################################################################

#                      二分查找   
################################################################


"""
35.搜索插入位置
给定一个排序数组和一个目标值,在数组中找到目标值,并返回其索引。如果目标值不存在于数组中,返回它将会被按顺序插入的位置。
输入: nums = [1,3,5,6], target = 5
输出: 2
"""
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: intm
        :rtype: int
        """
        # if target not in nums:
        #     nums.append(target)
        # nums.sort()
        # return nums.index(target)

        low= 0
        high = len(nums)

        while low < high:
            mid = low + (high-low) /2
            if nums[mid] > target:
                high = mid
            
            elif nums[mid] < target:
                low = mid + 1
            else:
                return mid
        
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
    # 由于每行的第一个元素大于前一行的最后一个元素,且每行元素是升序的,所以每行的第一个元素大于前一行的第一个元素,因此矩阵第一列的元素是升序的。我们可以对矩阵的第一列的元素二分查找,找到最后一个不大于目标值的元素,然后在该元素所在行中二分查找目标值是否存在。
        m, n = len(matrix), len(matrix[0])
        left, right = 0, m*n
        while left < right:
            mid = left + (right-left)//2
            x = matrix[mid//n][mid%n] # 定位几行几列的元素
            if x == target:
                return True
            elif x > target:
                right = mid # 缩小右侧边界
            else:
                left = mid + 1 # 缩小左侧边界
        return False

"""

33.搜索旋转排序数组
整数数组 nums 按升序排列,数组中的值 互不相同.
输入:nums = [4,5,6,7,0,1,2], target = 0
输出:4

"""
class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # if target in nums:
        #     return nums.index(target)
        # else:
        #     return -1

        # if len(nums) == 0:
        #     return -1
        # left,right = 0,len(nums)-1

        # while left < right:
        #     mid = left + (right-left) // 2

        #     if  nums[mid] < nums[right]:  # 升序
        #         if nums[mid] < target <= nums[right]:
        #             left = mid + 1
        #         else:
        #             right = mid
        #     else:
        #         if nums[left] <= target <= nums[mid]:
        #             right = mid
        #         else:
        #             left = mid + 1
            
        # return -1 if nums[left] != target else left
        if not nums:
            return -1

        l, r = 0, len(nums) - 1

        while l <= r:
            mid = l + (r-l)//2
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
            else:
                if nums[mid] < target <= nums[len(nums)-1]:
                    l = mid + 1
                else:
                    r = mid - 1
        return -1
    


# https://leetcode.cn/problems/number-of-islands/solutions/211211/dao-yu-lei-wen-ti-de-tong-yong-jie-fa-dfs-bian-li-
"""

200.岛屿数量
给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格,请你计算网格中岛屿的数量。
岛屿总是被水包围,并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
此外,你可以假设该网格的四条边均被水包围。
输入:grid = [
  ['1','1','1','1','0'],
  ['1','1','0','1','0'],
  ['1','1','0','0','0'],
  ['0','0','0','0','0']
]
输出:1

DFS 的基本结构
网格结构要比二叉树结构稍微复杂一些,它其实是一种简化版的图结构。要写好网格上的 DFS 遍历,我们首先要理解二叉树上的 DFS 遍历方法,再类比写出网格结构上的 DFS 遍历。我们写的二叉树 DFS 遍历一般是这样的:
def dfs(node):
    if node is None:
        return
    # 访问节点 node
    dfs(node.left)
    dfs(node.right)
这一点稍微有些反直觉,坐标竟然可以临时超出网格的范围？这种方法我称为「先污染后治理」—— 甭管当前是在哪个格子,先往四个方向走一步再说,如果发现走出了网格范围再赶紧返回。这跟二叉树的遍历方法是一样的,先递归调用,发现 root == null 再返回。

这样,我们得到了网格 DFS 遍历的框架代码:

def dfs(grid, i, j):
    # 终止条件
    if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]):
        return
    # 访问节点 grid[i][j]
    dfs(grid, i + 1, j)
    dfs(grid, i - 1, j)
    dfs(grid, i, j + 1)
    dfs(grid, i, j - 1)
接下来,我们来完成「访问节点 grid[i][j]」的代码。对于岛屿问题来说,访问节点的操作就是把当前格子标记为「已访问」,以免重复访问。我们可以把「1」改成「0」来表示这个格子已经被访问过了.

如何避免这样的重复遍历呢？答案是标记已经遍历过的格子。以岛屿问题为例,我们需要在所有值为 1 的陆地格子上做 DFS 遍历。每走过一个陆地格子,就把格子的值改为 2,这样当我们遇到 2 的时候,就知道这是遍历过的格子了。也就是说,每个格子可能取三个值:

0 —— 海洋格子
1 —— 陆地格子（未遍历过）
2 —— 陆地格子（已遍历过）

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        # 记录岛屿数量
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '1':
                    count += 1
                    # 从当前陆地格子开始 DFS 遍历
                    dfs(grid, i, j)
        return count

        def dfs(grid, i, j):
            # 终止条件
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] != '1':
                return
            # 标记为已遍历过
            grid[i][j] = '2'
            # 向四个方向进行 DFS 遍历
            dfs(grid, i + 1, j)
            dfs(grid, i - 1, j)
            dfs(grid, i, j + 1)
            dfs(grid, i, j - 1)
"""

#首先明确岛屿定义:四周环海的独立陆地或者四周环海的连续陆地(连续的定义是上下左右方向也存在陆地)
#深度遍历的网格搜索。类似于二叉树的深度遍历递归。
#二叉树的深度遍历搜索的base case通常是非叶子结点即返回,而网格深度遍历搜索的返回条件是当越界情况出现时
#二叉树的递归是其左右孩子结点,而网格的递归是上下左右四个方向。
#唯一不同之处在于:网格的递归需要将走过的路进行标记,不然上下左右四个方向存在“回头路”的情况
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        def dfs(i,j):
            # 边界条件, 
            if i <0 or i >= len(grid) or j < 0 or j>= len(grid[0]):
                return 
            # 终止条件
            if grid[i][j] != "1":
                return
            
            # 标记为已遍历过
            grid[i][j] = '2'
            # 四个方向都去探索,直至遇到终止条件
            dfs(i-1, j)
            dfs(i+1, j)
            dfs(i, j-1)
            dfs(i, j+1)
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    dfs(i,j)
                    count += 1
        return count



#################################################################

###                   栈与队列
#################################################################

"""
20. 有效的括号

给定一个只包括 '(',')','{','}','[',']' 的字符串 s ,判断字符串是否有效。

有效字符串需满足:

左括号必须用相同类型的右括号闭合。
左括号必须以正确的顺序闭合。
每个右括号都有一个对应的相同类型的左括号。

输入,s = "()"
输出,true

"""
class Solution:
    def isValid(self, s: str) -> bool:
        # 括号数量一定是偶数
        if len(s) % 2 == 1:
            return False
        # 括号对应关系
        pairs = {
            ")": "(",
            "]": "[",
            "}": "{",
        }
        # 栈的方式
        stack = list()
        for ch in s:
            # 如果是右括号,需要判断最左边这个是否是左括号,如果是则pop,不是则直接返回false
            # 或者右括号直接出现在第一个,则直接返回FALSE
            if ch in pairs:
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
            else:
                # 如果是左括号,则按顺序添加到栈中；
                stack.append(ch)
        
        return not stack


"""
155. 最小栈

设计一个支持 push ,pop ,top 操作,并能在常数时间内检索到最小元素的栈。

实现 MinStack 类:

MinStack() 初始化堆栈对象。
void push(int val) 将元素val推入堆栈。
void pop() 删除堆栈顶部的元素。
int top() 获取堆栈顶部的元素。
int getMin() 获取堆栈中的最小元素。
"""
class MinStack:

    def __init__(self):
        self.stack = []
        self.min_stack = [math.inf]
        

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(val)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()
        
"""394. 字符串解码
给定一个经过编码的字符串,返回它解码后的字符串.给定一个经过编码的字符串,返回它解码后的字符串。

编码规则为: k[encoded_string],表示其中方括号内部的 encoded_string 正好重复 k 次。注意 k 保证为正整数。"""

class Solution:
    def decodeString(self, s: str) -> str:
        stack, res, multi = [], "", 0
        for c in s:
            if c == '[':
                stack.append([multi, res]) # 记录下当前括号元素需要复制的次数
                res, multi = "", 0 # 重置res和multi
            elif c == ']':
                cur_multi, last_res = stack.pop() 
                res = last_res + cur_multi * res # 扩至
            elif '0' <= c <= '9':
                multi = multi * 10 + int(c) # 倍数
            else:
                res += c # 括号内元素
        return res

"""
739. 每日温度
给定一个整数数组 temperatures ,表示每天的温度,返回一个数组 answer ,其中 answer[i] 
是指对于第 i 天,下一个更高温度出现在几天后。如果气温在这之后都不会升高,请在该位置用 0 来代替。"""

class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        # res = []
        # for i, num in enumerate(temperatures[:-1]):
        #     flag = False
        #     for j in range(i+1, len(temperatures)):
        #         if temperatures[j] > temperatures[i]:
        #             res.append(j-i)
        #             flag = True
        #             break
        #     if not flag:
        #         res.append(0)
        # res.append(0)
        # return res
        ans = [0] * len(temperatures)
        stack = []
        for i in range(len(temperatures)):
            temperature = temperatures[i]
            # 栈不为空且当前温度大于栈顶温度时,弹出栈顶元素,并计算差值
            while stack and temperature > temperatures[stack[-1]]:
                prev_index = stack.pop()
                ans[prev_index] = i - prev_index
            # 记录当前温度的索引
            stack.append(i)
        return ans

from collections import defaultdict

"""
347. 前 K 个高频元素
给你一个整数数组 nums 和一个整数 k ,请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。
输入:nums = [1,1,1,2,2,3], k = 2
输出:[1,2]
"""
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        return [num for num, _ in Counter(nums).most_common(k)]
    

#################################################################
###                   贪心算法
#################################################################


"""
121. 买卖股票的最佳时机
给定一个数组 prices ,它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。

你只能选择 某一天 买入这只股票,并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润

股票问题的方法就是 动态规划,因为它包含了重叠子问题,即买卖股票的最佳时机是由之前买或不买的状态决定的,而之前买或不买又由更早的状态决定的...
"""


"""
方法二:动态规划
动态规划一般分为一维、二维、多维（使用 状态压缩）,对应形式为 dp(i)、dp(i)(j)、二进制dp(i)(j)。

1. 动态规划做题步骤

明确 dp(i) 应该表示什么（二维情况:dp(i)(j)）；
根据 dp(i) 和 dp(i−1) 的关系得出状态转移方程；
确定初始条件,如 dp(0)。
2. 本题思路

其实方法一的思路不是凭空想象的,而是由动态规划的思想演变而来。这里介绍一维动态规划思想。

dp[i] 表示前 i 天的最大利润,因为我们始终要使利润最大化,则:

dp[i]=max(dp[i−1],prices[i]−minprice)

"""
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # max_profit = 0
        # min_price = float('inf')  # 初始化为正无穷大
        # for price in prices:
        #     min_price = min(min_price, price)  # 更新历史最低价格
        #     max_profit = max(max_profit, price - min_price)  # 计算最大利润
        # return max_profit

        dp = [0] * len(prices)  # 初始化dp数组,dp[0]=0,即第0天利润为0
        min_price = prices[0]
        for i in range(1, len(prices)):
            min_price = min(min_price, prices[i]) # 更新历史最低价格
            dp[i] = max(dp[i-1], prices[i] - min_price) # 计算第i天的最大利润
        return dp[-1]

"""

55. 跳跃游戏
给你一个非负整数数组 nums ,你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标,如果可以,返回 true ；否则,返回 false 。

输入:nums = [2,3,1,1,4]
输出:true  """
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        right_most = 0 #记录最远能跳到哪
        for idx, val in enumerate(nums[:-1]): # 忽略最后一个位置
            right_most = max(right_most, idx + val) # 当前位置上,最远能跳到哪
            # 如果最远都无法超过当前位置,那肯定无法到达最后一个位置,提前结束
            if right_most <= idx:
                return False
        return True
    
"""
45. 跳跃游戏 II
给定一个长度为 n 的 0 索引整数数组 nums。初始位置在下标 0。
每个元素 nums[i] 表示从索引 i 向后跳转的最大长度。换句话说,如果你在索引 i 处,你可以跳转到任意 (i + j) 处:
返回到达 n - 1 的最小跳跃次数。测试用例保证可以到达 n - 1。
输入: nums = [2,3,0,1,4]
输出: 2
"""
class Solution:
    def jump(self, nums: List[int]) -> int:
        # 贪心算法,反向出发查找位置
        position = len(nums) - 1
        steps = 0 # 记录步数
        while position > 0:
            for i in range(position):
                if i + nums[i] >= position: # 如果当前位置+跳数>=目标位置
                    position = i # 更新目标位置为当前位置
                    steps += 1 # 步数+1
                    break
        return steps
    



"""763. 划分字母区间
字符串 s 由小写英文字母组成。我们要把这个字符串划分为尽可能多的片段,同一字母最多出现在一个片段中。返回每个字符串片段的长度组成的列表。
输入:s = "ababcbacadefegdehijhklij"
输出:[9,7,8]
解释: 划分结果为 "ababcbaca", "defegde", "hijhklij"。
每个字母最多出现在一个片段中。
"""

class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # 创建一个字典 last,用来记录每个字符在字符串中的最后出现位置
        last = {}
        # 遍历字符串 s,记录每个字符最后出现的位置
        for i, ch in enumerate(s):
            last[ch] = i
        # 初始化 partition 列表,用于存储每个分割部分的长度
        partition = []
        # start 和 end 表示当前分区的开始和结束位置
        start = end = 0

        # 遍历字符串中的每个字符
        for i, ch in enumerate(s):
            # 更新当前字符的最后位置（`end` 记录当前分区能扩展到的最远位置）
            end = max(end, last[ch])
            
            # 如果当前位置 i 和当前分区的结束位置 `end` 相同,说明找到一个合法分割
            if i == end:
                # 记录当前分区的长度（`end - start + 1`）
                partition.append(end - start + 1)
                # 更新 start 为下一个分区的开始位置
                start = end + 1
        
        return partition



##################################################################
###                  动态规划
############################################################    ############



"""
70. 爬楼梯
假设你正在爬楼梯。需要 n 阶才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？
输入:n = 3
输出:3
解释:有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
"""
class Solution:
    def climbStairs(self, n: int) -> int:
        # 它意味着爬到第 x 级台阶的方案数是爬到第 x−1 级台阶的方案数和爬到第 x−2 级台阶的方案数的和。
        # f(x) = f(x-1)+f(x-2)
        dp = [0,1,2] # 初始化dp数组
        i = 3 # 从3开始
        while i <= n: # 循环到n
            dp.append(dp[i-1]+dp[i-2]) # 状态转移方程
            i += 1 # 递增
        return dp[n] # 返回dp[n]
    
"""
118. 杨辉三角
给定一个非负整数 numRows,生成「杨辉三角」的前 numRows 行。
在「杨辉三角」中,每个数是它左上方和右上方的数相加
输入: numRows = 5
输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]

每个数字等于上一行的左右两个数字之和,可用此性质写出整个杨辉三角。即第 n 行的第 i 个数等于第 n−1 行的第 i−1 个数和第 i 个数之和。
"""
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        result = []
        for i in range(numRows):
            # row = [None for _ in range(i + 1)] # 初始化行,长度为i+1
            # row[0], row[-1] = 1, 1 # 首尾赋值1
            row = [1] * (i + 1) # 初始化行,首尾赋值1
            for j in range(1, len(row) - 1): # 中间赋值,从1到len(row)-2
                row[j] = result[i - 1][j - 1] + result[i - 1][j] # 状态转移方程,第j个数等于上一行的第j-1个数和第j个数之和
            result.append(row)
        return result
    

"""
198.打劫房屋
你是一个专业的小偷,计划偷窃沿街的房屋。每间房内都藏有一定的现金,影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统,如果两间相邻的房屋在同一
晚上被小偷闯入,系统会自动报警。
给定一个代表每个房屋存放金额的非负整数数组,计算你 不触动警报装置的情况下 ,一夜之内能够偷窃到的最高金额。
输入:[1,2,3,1]
输出:4
解释:偷窃 1 号房屋 (金额 = 1) ,然后偷窃 3 号房屋 (金额 = 3)。
     偷窃到的最高金额 = 1 + 3 = 4 。
     dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
"""
class Solution:
    def rob(self, nums: List[int]) -> int:
        if not nums:
            return 0
        if len(nums) <= 2:
            return max(nums)
        dp = [0] * len(nums) # 初始化dp数组, 表示到第i个房子时,能偷到的最大金额
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1]) # 初始化dp[0]和dp[1],表示前两个房子能偷到的最大金额
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]) # 状态转移方程,即第i个房子能偷到的最大金额,等于前两个房子能偷到的最大金额中的较大者加上当前房子能偷到的金额
        return dp[-1]
    
"""

279. 完全平方数
给你一个整数 n ,返回 和为 n 的完全平方数的最少数量 。完全平方数 是一个整数,其值等于另一个整数的平方 ;换句话说,它就是某个整数的平方。
输入:n = 12
输出:3
解释:12 = 4 + 4 + 4
dp[i] = min(dp[i], dp[i - j*j] + 1)
"""
class Solution:
    def numSquares(self, n: int) -> int:
        # dp[i]表示组成数字i所需的最少完全平方数数量
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # 组成数字0需要0个完全平方数

        # 预先计算所有小于等于n的完全平方数
        squares = []
        for i in range(1, int(n**0.5) + 1):
            squares.append(i * i)

        # 填充dp数组
        for i in range(1, n + 1):
            for square in squares:
                if i < square: # 如果平方数大于i,则不需要继续计算
                    break
                dp[i] = min(dp[i], dp[i - square] + 1)  # 状态转移方程,即组成数字i所需的最少完全平方数数量,等于组成数字i-square所需的最少完全平方数数量加上1

        return dp[n]

"""

322.零钱兑换
给你一个整数数组coins ,表示不同面额的硬币;以及一个整数amount ,表示总金额。计算并返回可以凑成总金额所需的 最少的硬币个数 。如果没有任何一种硬币组合能组成总金额,返回 -1 。
你可以认为每种硬币的数量是无限的。
输入:coins = [1,2,5], amount = 11
输出:3
解释:11 = 5 + 5 + 1
"""

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp[i]表示组成金额i所需的最少硬币数量
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0  # 组成金额0需要0个硬币

        # 填充dp数组
        for i in range(1, amount + 1):
            for coin in coins:
                if i < coin: # 如果硬币面值大于i,则不需要继续计算
                    continue
                dp[i] = min(dp[i], dp[i - coin] + 1)  # 状态转移方程,即组成金额i所需的最少硬币数量,等于组成金额i-coin所需的最少硬币数量加上1

        return dp[amount] if dp[amount] != float('inf') else -1


"""139. 单词拆分
给你一个字符串 s 和一个字符串列表 wordDict 作为字典,判定 s 是否可以由空格拆分为一个或多个在字典中出现的单词。
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分为 "leet code"
"""

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        dp = [False] * (len(s) + 1) # 初始化dp数组,表示s[:i]能否被拆分成wordDict中的单词
        dp[0] = True  # 空字符串""能被拆分成wordDict中的单词

        # 填充dp数组
        for i in range(1, len(s) + 1):
            for word in wordDict:
                if i >= len(word) and s[i - len(word):i] == word:  # 如果s[i-len(word):i]与word匹配
                    dp[i] |= dp[i - len(word)]  # 状态转移方程,即s[:i]能否被拆分成wordDict中的单词,取决于s[:i-len(word)]能否被拆分成wordDict中的单词

        return dp[len(s)]


"""300. 最长递增子序列
给你一个整数数组 nums ,找到其中最长严格递增子序列的长度。
输入:nums = [10,9,2,5,3,7,101,18]
输出:4
解释:最长递增子序列是 [2,3,7,101],因此长度为 4 。
"""

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n  # 初始化dp数组,表示以nums[i]结尾的最长递增子序列长度至少为1

        # 填充dp数组
        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:  # 如果nums[i]大于nums[j],则可以将nums[i]加入到以nums[j]结尾的最长递增子序列中
                    dp[i] = max(dp[i], dp[j] + 1)  # 状态转移方程,即以nums[i]结尾的最长递增子序列长度,等于以nums[j]结尾的最长递增子序列长度加上1

        return max(dp)  # 返回dp数组中的最大值,即整个数组的最长递增子序列长度



"""152. 乘积最大子数组
给你一个整数数组 nums ,请你找出数组中乘积最大的连续子数组
输入:nums = [2,3,-2,4]
输出:6
解释:子数组 [2,3] 有最大乘积 6 。
"""

class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        max_product = nums[0] # 初始化最大乘积为第一个元素
        current_min = nums[0] # 初始化当前最小值为第一个元素
        current_max = nums[0] # 初始化当前最大值为第一个元素

        for num in nums[1:]:
            prev_min = current_min # 记录前一个最小值
            prev_max = current_max # 记录前一个最大值

            current_min = min(prev_min * num, prev_max * num, num) # 状态转移方程,即当前最小值,等于前一个最小值乘以当前元素,前一个最大值乘以当前元素,当前元素中的最小值
            current_max = max(prev_min * num, prev_max * num, num) # 状态转移方程,即当前最大值,等于前一个最小值乘以当前元素,前一个最大值乘以当前元素,当前元素中的最大值

            max_product = max(max_product, current_max) # 更新最大乘积

        return max_product



""" 416. 分割等和子集
给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集,使得两个子集的元素和相等。
输入:nums = [1,5,11,5]
输出:true   

方案一: 0-1 背包问题
我们可以将数组划分为两个子集,使得两个子集的元素和相等,等价于在数组中找到一个子集,使得该子集的元素和等于数组元素和的一半。
因此,我们可以将问题转化为一个 0-1 背包问题,即在数组中选择一些元素,使得这些元素的和等于目标值(数组元素和的一半)。
我们可以使用动态规划来解决这个问题。我们定义一个二维数组 dp,其中 dp[i][j] 表示在前 i 个元素中,是否存在一个子集,使得这些元素的和等于 j。
状态转移方程为:            dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i-1]]    if j-nums[i-1] >= 0
"""

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total_sum = sum(nums)
        if total_sum % 2 == 1: # 如果总和为奇数,则不可能划分为两个相等的子集
            return False
        target = total_sum // 2
        dp = [True] + [False] * target # 初始化dp数组,表示是否能组成目标和target,dp[0]=True,表示能组成目标和t
        for num in nums:
            for i in range(target, num - 1, -1): # 从target开始,每次减去当前元素,直到小于当前元素
                dp[i] = dp[i] or dp[i - num] # 状态转移方程,即能否组成目标和i,取决于能否组成目标和i-num或者能否组成目标和i
        return dp[target]


"""32. 最长有效括号
给你一个只包含 '(' 和 ')' 的字符串,找出最长有效(格式正确且连续)括号子串的长度。
输入:s = "(()"
输出:2
"""


class Solution:
    def longestValidParentheses(self, s: str) -> int:
        stack = [-1] # 初始化栈,栈底元素为-1,表示当前未匹配的左括号的下标
        res = 0
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i) # 遇到左括号,将其下标入栈
            else:
                stack.pop() # 遇到右括号,弹出栈顶元素,即匹配的左括号的下标
                if not stack: # 如果栈为空,说明当前右括号没有匹配的左括号,将其下标入栈
                    stack.append(i) # 将当前右括号的下标入栈
                else:
                    res = max(res, i - stack[-1]) # 计算当前匹配的括号长度,更新最大值
        return res


##################################################################
###                  多维动态规划
##################################################################

"""
# 62. 不同路径
给你一个 m x n 的网格,请你计算从左上角到右下角有多少条不同的路径。
输入:m = 3, n = 7
输出:28

dp[i][j] = dp[i-1][j] + dp[i][j-1]
"""

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0] * n for _ in range(m)]  # 初始化dp数组,表示从(0,0)到(i,j)的路径数
        for i in range(m):
            dp[i][0] = 1 # 初始化第一列,从(0,0)到(i,0)的路径数为1
        for j in range(n):
            dp[0][j] = 1 # 初始化第一行,从(0,0)到(0,j)的路径数为1
        for i in range(1, m): # 填充其他位置
            for j in range(1, n): 
                # 状态转移方程,即从(0,0)到(i,j)的路径数,等于从(0,0)到(i-1,j)的路径数加上从(0,0)到(i,j-1)
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]
        

"""
# 64. 最小路径和
给定一个包含非负整数的 m x n 网格 grid ,请找出一条从左上角到右下角的路径,使得路径上的数字总和为最小。
输入:grid = [[1,3,1],[1,5,1],[4,2,1]]
输出:7
解释:因为路径 1→3→1→1→1 的总和最小。
dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
"""

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        # dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        m, n = len(grid), len(grid[0])
        dp = [[0] * n for _ in range(m)] # 初始化dp数组,表示从(0,0)到(i,j)的最小路径和
        dp[0][0] = grid[0][0] # 初始化起点
        # 初始化第一行和第一列
        for i in range(1, m):
            dp[i][0] = dp[i - 1][0] + grid[i][0] # 第一列只能从上方走下来
        for j in range(1, n): 
            dp[0][j] = dp[0][j - 1] + grid[0][j] # 第一行只能从左方走过来
        for i in range(1, m): # 填充其他位置
            for j in range(1, n): 
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j] # 状态转移方程
        return dp[m - 1][n - 1]



"""5. 最长回文子串
给你一个字符串 s,找到 s 中最长的回文子串。
输入:s = "babad"
输出:"bab"
解释:"aba" 也是一个有效答案。
"""

# dp[i][j] = (s[i] == s[j]) and (j - i < 3 or dp[i + 1][j - 1])

class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        if n < 2:
            return s
        start, max_len = 0, 1 # 初始化最大长度为1,起始位置为0
        dp = [[False] * n for _ in range(n)]  # 初始化dp数组,表示s[i:j+1]是否为回文串
        for j in range(1, n): 
            for i in range(j): 
                if s[i] == s[j]:  # 如果s[i]和s[j]相等
                    if j - i < 3:  # 如果子串长度小于3,则一定是回文串
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i + 1][j - 1]  # 状态转移方程,即s[i:j+1]是否为回文串,取决于s[i+1:j-1]是否为回文串
                if dp[i][j] and j - i + 1 > max_len:  # 如果s[i:j+1]是回文串且长度大于max_len
                    max_len = j - i + 1  # 更新最大长度
                    start = i  # 更新起始位置
        return s[start:start + max_len]


"""1143. 最长公共子序列
给定两个字符串 text1 和 text2,返回这两个字符串的最长公共子序列的长度。
输入:text1 = "abcde", text2 = "ace"
输出:3
解释:最长公共子序列是 "ace" ,它的长度为 3 。
复杂度分析:
时间复杂度:O(mn),其中 m 和 n 分别是字符串 text1 和 text2 的长度。我们需要填充一个大小为 m x n 的二维数组 dp,每个位置的计算需要 O(1) 的时间。
空间复杂度:O(mn),我们使用了一个大小为 m x n 的二维数组 dp 来存储中间结果。
"""

class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化dp数组,表示text1的前i个字符与text2的前j个字符的LCS长度
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:  # 如果text1[i-1]和text2[j-1]相等
                    dp[i][j] = dp[i - 1][j - 1] + 1  # 状态转移方程,即text1的前i个字符与text2的前j个字符的LCS长度,等于text1的前i-1个字符与text2的前j-1个字符的LCS长度加1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])  # 状态转移方程,即text1的前i个字符与text2的前j个字符的LCS长度,等于text1的前i-1个字符与text2的前j个字符的LCS长度和text1的前i个字符与text2的前j-1个字符的LCS长度中的较大值
        return dp[m][n]  # 返回text1的前m个字符与text2的前n个字符的LCS长度


"""72. 编辑距离
给你两个单词 word1 和 word2,请你计算出将 word1 转换成 word2 所使用的最少操作数 。
你可以对一个单词进行如下三种操作:
插入一个字符
删除一个字符
替换一个字符
输入:word1 = "horse", word2 = "ros"
输出:3
dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1]) and (word1[i-1] != word2[j-1])
"""

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化dp数组,表示word1的前i个字符与word2的前j个字符的编辑距离
        
        for i in range(m + 1):
            dp[0][i] = i  # 初始化第一列,将word1的前i个字符转换为空字符串需要i次删除操作
        for j in range(n + 1):
            dp[j][0] = j  # 初始化第一行,将空字符串转换为word2的前j个字符需要j次插入操作
        
        for i in range(1, m + 1):     
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:  # 如果word1[i-1]和word2[j-1]相等
                    dp[i][j] = dp[i - 1][j - 1]  # 状态转移方程,即word1的前i个字符与word2的前j个字符的编辑距离,等于word1的前i-1个字符与word2的前j-1个字符的编辑距离
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1,    # 删除操作
                                   dp[i][j - 1] + 1,    # 插入操作
                                   dp[i - 1][j - 1] + 1) # 替换操作
        return dp[m][n]  # 返回word1的前m个字符与word2的前n个字符的编辑距离



#################################################################
###                  区间动态规划
#################################################################


"""136. 只出现一次的数字
给你一个非空整数数组 nums ,除某个元素只出现一次外,其余每个元素均出现两次。找出那个只出现了一次的元素。
输入:nums = [4,1,2,1,2]
输出:4
"""
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # result = 0
        # for num in nums:
        #     result ^= num  # 使用异或运算,相同的数字异或结果为0,不同的数字异或结果为1
        # return result
        for i in nums:
            if nums.count(i) == 1:
                return i

"""169. 多数元素
给定一个大小为 n 的数组 nums ,返回其中的多数元素。多数元素是指在数组中出现次数 大于 ⌊ n/2 ⌋ 的元素。你可以假设数组是非空的,并且给定的数组总是存在多数元素。
输入:nums = [3,2,3]
输出:3
"""

class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # for i in nums:
        #     if nums.count(i) > len(nums)/2:
        #         return i
        nums.sort()
        return nums[len(nums)//2]
    
"""75. 颜色分类
给定一个包含红色、白色和蓝色,即元素为 0、 1 和 2 的数组,对它们进行排序,使得相同颜色的元素相邻,并按照 0、1、2 的顺序排列。
输入:nums = [2,0,2,1,1,0]
输出:[0,0,1,1,2,2]
"""

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        count = [0, 0, 0]  # 计数器,分别记录0,1,2的数量
        for num in nums:
            count[num] += 1  # 统计每个数字的数量
        index = 0
        for i in range(3):  # 遍历0,1,2
            for _ in range(count[i]):  # 根据数量填充nums数组
                nums[index] = i
                index += 1

"""31. 下一个排列
实现获取下一个排列的函数,算法需要将给定数字序列重新排列成字典序中下一个更大的排列。 如果不存在下一个更大的排列,则将数字重新排列成最小的排列(即升序排列)。
必须 原地 修改只允许使用额外常数空间。
输入:nums = [1,2,3]
输出:[1,3,2]
"""

class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        for i in range(len(nums)-1, 0, -1):
            if nums[i] > nums[i-1]:
                nums[i:] = sorted(nums[i:])  # 将后面的元素排序
                for j in range(i, len(nums)):  # 从i开始遍历到末尾
                    if nums[j] > nums[i-1]: # 找到第一个大于nums[i-1]的元素
                        nums[j], nums[i-1] = nums[i-1], nums[j] # 交换i-1和j位置的元素
                        break
                return
        nums.sort()
        # for i in range(len(nums)-1, 0, -1):
        #     if nums[i] > nums[i-1]:
        #         break
        # else:
        #     nums.reverse()
        #     return
        # for j in range(len(nums)-1, i-1, -1):
        #     if nums[j] > nums[i-1]:
        #         nums[j], nums[i-1] = nums[i-1], nums[j]
        #         break
        # nums[i:] = reversed(nums[i:])


"""

287. 寻找重复数
给定一个包含 n + 1 个整数的数组 nums ,其数字都在 [1, n] 范围内（包括 1 和 n）,可知至少存在一个重复的整数。假设只有一个重复的整数,找出这个重复的数。
输入:nums = [1,3,4,2,2]
输出:2 """

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        for i in range(len(nums)):
            if nums[abs(nums[i])] > 0:
                nums[abs(nums[i])] = -nums[abs(nums[i])]
            else:
                return abs(nums[i])




```
