####################################
#    双指针
####################################


"""
283.移动零
给定一个数组 nums,编写一个函数将所有 0 移动到数组的末尾,同时保持非零元素的相对顺序。
输入:nums = [0,1,0,3,12]
输出:[1,3,12,0,0]
# 双指针法
时间复杂度:O(n),空间复杂度:O(1)
"""
# 双指针法,时间复杂度O(n),空间复杂度O(1)
# 方法:使用一个指针size记录非零元素的个数,遍历数组,将非零元素移动到前面,最后将剩余位置补零
class Solution:
    def moveZeroes(self, nums):
        size = 0
        # 遍历数组,将非零元素移动到前面,同时记录非零元素的个数
        for x in nums:
            if x:
                nums[size] = x
                size += 1
        # 将数组剩余位置补零
        for i in range(size, len(nums)):
            nums[i] = 0


"""
11.盛最多水的容器
给你 n 个非负整数 a1,a2,...,an,每个数代表坐标中的一个点 (i,ai) 。在坐标内画 n 条垂直线,其中第 i 条线的两个端点分别为 (i,ai) 和 (i,0)。找出其中的两条线,使得它们与 x 轴共同构成的容器可以容纳最多的水。
输入:[1,8,6,2,5,4,8,3,7]
输出:49
# 双指针法
时间复杂度:O(n),空间复杂度:O(1)"""
# 双指针法,时间复杂度O(n),空间复杂度O(1)
# 方法:使用左右指针分别指向数组的两端,计算当前面积,更新最大面积,然后移动较短的指针向中间移动,因为较短的指针决定了当前面积的大小,移动较短的指针可能会增加面积,而移动较长的指针不可能增加面积
class Solution:
    def maxArea(self, height):
        # 初始化左右指针,最大面积为0a
        left, right = 0, len(height) - 1
        max_area = 0
        while left < right:
            # 计算当前面积
            area = (right - left) * min(height[left], height[right])
            max_area = max(max_area, area)
            # 移动较短的指针
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_area



"""
42.接雨水
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图,计算按此排列的柱子,下雨之后能接多少雨水。
输入:height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出:6
"""
# 双指针法,时间复杂度O(n),空间复杂度O(1)
# 方法i:使用左右指针分别指向数组的两端,并维护左右两侧的最大高度leftMax和rightMax。
# 每次比较height[left]和height[right]的大小,将较小的一侧的指针向中间移动,并计算当前位置的储水量。
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        left, right = 0, len(height) - 1
        leftMax = rightMax = 0

        while left < right:
            # 如果 height[left] < height[right], 说明左边的最大高度 leftMax 决定了当前位置的储水量。计算 leftMax - height[left] 并加到 ans 中,然后 left 右移。
            leftMax = max(leftMax, height[left])
            rightMax = max(rightMax, height[right])
            # 如果 height[left] < height[right],则 leftMax 是 left 位置的约束（因为右边有更高的柱子 height[right] 挡着,水的高度不会超过 leftMax）。
            if height[left] < height[right]:
                ans += leftMax - height[left] # 计算 left 位置的储水量并加到 ans 中
                left += 1 # 左移left指针
            else:
                ans += rightMax - height[right] # 计算 right 位置的储水量并加到 ans 中
                right -= 1 # 右移right指针
        return ans
    


"""
15.三数之和
给你一个整数数组 nums ,判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ,
同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。注意：答案中不可以包含重复的三元组。
输入:nums = [-1,0,1,2,-1,-4]
输出:[[-1,-1,2],[-1,0,1]]
"""
# 双指针法,时间复杂度O(n^2),空间复杂度O(1)
# 方法:先对数组进行排序,然后遍历数组,固定第一个元素,使用双指针寻找剩余元素,并跳过左右重复元素
class Solution:
    def threeSum(self, nums):
        res = []
        nums.sort()  # 先排序
        n = len(nums)
        # 遍历数组,固定第一个元素,然后使用双指针寻找剩余元素
        for i in range(n-2):
            if i > 0 and nums[i] == nums[i-1]:  # 跳过重复元素
                continue
            left, right = i + 1, n - 1  # 双指针
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                # 判断三数之和是否为0,并移动指针
                if total == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left + 1]:  # 跳过重复元素
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:  # 跳过重复元素
                        right -= 1
                    left += 1
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1
        return res
    



####################################################
#    滑动窗口
####################################################

"""
3.无重复字符的最长子串
给定一个字符串 s ,请你找出其中不含有重复字符的 最长子串 的长度。
输入:s = "abcabcbb"
输出:3
"""
# 滑动窗口法, 时间复杂度O(n),空间复杂度O(min(m,n)),m为字符集大小
# 方法:使用滑动窗口,不断地扩大窗口,直到窗口内的字符串没有重复字符,此时窗口的大小就是最长的子串的长度
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        char_set = set()
        left = 0
        max_length = 0
        for right in range(len(s)):
            # 如果当前字符已经在窗口中,则将窗口左边界向右移动,直到窗口中没有重复字符
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            char_set.add(s[right])
            max_length = max(max_length, right - left + 1) # 更新最大长度
        return max_length


"""
438.找到字符串中所有字母异位词
给定两个字符串 s 和 p,找到 s 中所有 p 的 异位词 的子串,返回这些子串的起始索引。不考虑答案输出的顺序。
输入:s = "cbaebabacd", p = "abc"
输出:[0,6]
"""
# 滑动窗口法,时间复杂度O(n),空间复杂度O(1)
# 方法:1.使用滑动窗口,维护一个窗口内的字符频率,当窗口内的字符频率与p的字符频率相同时,
# 2.说明找到了一个异位词,记录窗口的起始索引,然后继续移动窗口,直到遍历完整个字符串s
class Solution:
    def findAnagrams(self, s: str, p: str):
        from collections import Counter
        p_count = Counter(p)  # 统计p中字符的频率
        s_count = Counter()   # 统计当前窗口中字符的频率
        res = []
        left = 0
        for right in range(len(s)):
            s_count[s[right]] += 1  # 将当前字符加入窗口
            # 3.当窗口大小超过p的长度时,将左边界向右移动,并更新窗口内字符的频率
            if right - left + 1 > len(p):
                s_count[s[left]] -= 1
                if s_count[s[left]] == 0:
                    del s_count[s[left]]
                left += 1
            # 当窗口内字符的频率与p的字符频率相同时,说明找到了一个异位词,记录窗口的起始索引
            if s_count == p_count:
                res.append(left)
        return res



#########################################
# 子串
##########################################

"""
560.和为 K 的子数组
给你一个整数数组 nums 和一个整数 k ,请你统计并返回 该数组中和为 k 的子数组的个数 。
子数组是数组中元素的连续非空序列。
输入:nums = [1,1,1], k = 2
输出:2
"""
# 前缀和 + 哈希表,时间复杂度O(n),空间复杂度O(n)
# 方法:使用前缀和数组prefix_sum,其中prefix_sum[i]表示数组nums中前i个元素的和。
# 对于每个prefix_sum[i],我们需要找到一个prefix_sum[j],使得prefix_sum[i] - prefix_sum[j] = k,
# 即prefix_sum[j] = prefix_sum[i] - k。

# 我们可以使用一个哈希表prefix_sum_count来记录每个前缀和出现的次数,初始时prefix_sum_count[0] = 1,表示前缀和为0的情况。
# 当我们遍历数组nums时,计算当前的前缀和prefix_sum,然后检查哈希表中是否存在prefix_sum - k,如果存在,则说明找到了一个和为k的子数组,将其出现次数加到结果中。
class Solution:
    def subarraySum(self, nums, k):
        from collections import defaultdict
        prefix_sum_count = defaultdict(int)
        prefix_sum_count[0] = 1
        prefix_sum = 0
        count = 0
        for num in nums:
            # 计算当前前缀和
            prefix_sum += num
            # 如果存在一个prefix_sum[j]使得prefix_sum[i] - prefix_sum[j] = k,则说明找到了一个和为k的子数组
            count += prefix_sum_count[prefix_sum - k]
            # 将当前前缀和加入哈希表中,记录出现的次数,以便后续查找
            prefix_sum_count[prefix_sum] += 1
            print(prefix_sum_count)
        return count


"""
239.滑动窗口最大值
给你一个整数数组 nums,有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。
返回 滑动窗口中的最大值 。
输入:nums = [1,3,-1,-3,5,3,6,7], k = 3
输出:[3,3,5,5,6,7]
"""
# 双端队列,时间复杂度O(n),空间复杂度O(k)
# 方法:使用一个双端队列deque来存储当前窗口内的元素索引,保证队列中的元素索引对应的值是递减的。
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums, k):
        dq = deque()
        result = []
        for i in range(len(nums)):
            # 1.如果队列不为空且队尾元素对应的值小于等于当前元素,则弹出队尾元素
            while dq and nums[dq[-1]] <= nums[i]:
                dq.pop()
            # 2.将当前元素的索引添加到队列末尾
            dq.append(i)
            # 3.如果队首元素不在窗口范围内,则将其弹出
            if dq[0] + k <= i:
                dq.popleft()
            # 如果窗口已经形成,则将队首元素对应的值添加到结果集中
            if i >= k - 1:
                result.append(nums[dq[0]])
        return result


"""
# 76.最小覆盖子串
给定两个字符串 s 和 t,长度分别是 m 和 n,返回 s 中的 最短窗口 子串,使得该子串包含 t 中的每一个字符（包括重复字符）。
如果没有这样的子串,返回空字符串 ""。
输入:s = "ADOBECODEBANC", t = "ABC"
输出:"BANC"
"""
# 滑动窗口法,时间复杂度O(m+n),空间复杂度O(m+n)
# 方法:使用滑动窗口,维护一个窗口内的字符频率,当窗口内的字符频率满足t的字符频率时, 说明找到了一个覆盖子串,
# 记录窗口的起始索引和长度,然后继续移动窗口,直到遍历完整个字符串s

# 这道题与438题类似,都是使用滑动窗口来维护一个窗口内的字符频率,但是这道题需要满足窗口内的字符频率大于等于t的字符频率,而438题需要满足窗口内的字符频率等于p的字符频率。
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        cnt_s = Counter()  # s 子串字母的出现次数
        cnt_t = Counter(t)  # t 中字母的出现次数

        ans_left, ans_right = -1, len(s) # 记录最短覆盖子串的左右端点,初始值为无效范围
        left = 0
        for right, c in enumerate(s):  # 移动子串右端点
            cnt_s[c] += 1  # 右端点字母移入子串
            while cnt_s >= cnt_t:  # 涵盖
                if right - left < ans_right - ans_left:  # 找到更短的子串
                    ans_left, ans_right = left, right  # 记录此时的左右端点
                cnt_s[s[left]] -= 1  # 左端点字母移出子串
                left += 1
        return "" if ans_left < 0 else s[ans_left: ans_right + 1]
