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

        # dp[i]表示和为i的完全平方数的最少数量
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        squares = []
        for i in range(1, int(n**0.5) + 1):
            squares.append(i * i)
        
        # 填充dp数组
        for i in range(1, n+1):
            for square in squares:
                if i < square:
                    break
                dp[i] = min(dp[i], dp[i-square]+ 1)

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
        # dp[i]表示凑成总金额i所需的最少硬币个数
        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for i in range(1, amount + 1):
            for coin in coins:
                if i < coin:
                    continue
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1


"""139. 单词拆分
给你一个字符串 s 和一个字符串列表 wordDict 作为字典,判定 s 是否可以由空格拆分为一个或多个在字典中出现的单词。
输入: s = "leetcode", wordDict = ["leet", "code"]
输出: true
解释: 返回 true 因为 "leetcode" 可以被拆分为 "leet code"
"""

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        # dp[i]表示s[0:i]是否可以被拆分为字典中的单词
        dp = [False] * (len(s) + 1)
        dp[0] = True # 空字符串可以被拆分为空字符串
        for i in range(1, len(s)+1):
            for word in wordDict:
                if i >= len(word) and s[i-len(word):i] == word:
                    dp[i] |=  dp[i - len(word)]
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

        for i in range(1, n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)  # 状态转移方程,即以nums[i]结尾的最长递增子序列长度等于以nums[j]结尾的最长递增子序列长度加1

        return max(dp)  # 返回dp数组中的最大值,即为整个数组的最长递增子序列长度


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
        total = sum(nums)
        if total %2 != 0:
            retrun False
        # dp[i]表示是否存在一个子集,使得这些元素的和等于i
        target = total // 2
        dp = [False] * (target + 1)
        dp[0] = True
        
        for num in nums:
            for j in range(target, num - 1, -1):
                dp[i] = dp[i] or dp[i - num] # 状态转移方程,即在前i个元素中,是否存在一个子集,使得这些元素的和等于j
        return dp[target]


"""32. 最长有效括号
给你一个只包含 '(' 和 ')' 的字符串,找出最长有效(格式正确且连续)括号子串的长度。
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
                stack.pop()
                if not stack:
                    stack.append(i) # 栈为空,将当前索引入栈
                else:
                    res = max(res, i-stack[-1]) # 计算当前匹配的括号长度,更新最大值
        return res


# 62. 不同路径
# 给你一个 m x n 的网格,请你计算从左上角到右下角有多少条不同的路径。
# 输入:m = 3, n = 7
# 输出:28

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # dp[i][j] 表示到达位置(i,j)的不同路径数, dp[i][j] = dp[i-1][j] + dp[i][j-1]
        dp = [[0]*n for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        
        for j in range(n):
            dp[0][j] = 1
        
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]

# 64. 最小路径和
# 给定一个包含非负整数的 m x n 网格 grid ,请找出一条从左上角到右下角的路径,使得路径上的数字总和为最小。
# 输入:grid = [[1,3,1],[1,5,1],[4,2,1]]
# 输出:7
# 解释:因为路径 1→3→1→1→1 的总和最小。
# dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        # dp[i][j] 表示到达位置(i,j)的最小路径和, dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j])
        m, n = len(grid), len(grid[0])
        dp = [[0]*n for _ in range(m)]
        dp[0][0] = grid[0][0]

        for i in range(1, m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1, n):
            dp[0][j] = dp[0][j-1] + grid[0][j]

        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        
        return dp[m-1][n-1]




# 最长回文子串
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # dp[i][j]表示s[i:j+1]为回文字符串 dp[i][j] = True if s[i]==s[j] and dp[i+1][j-1] else False
        dp = [[False]*len(s) for _ in range(len(s))]
        dp[0][0] = True
        start, max_len = 0, 1
        n = len(s)
        for i in range(1, n):
            for j in range(i):
                if s[i] == s[j] 
                    if i - j <3:
                        dp[j][i] = True
                    else:
                        dp[j][i] = dp[j+1][i-1]
                if dp[j][i] and i -j +1 > max_len:
                    max_len = i - j +1
                    start = j
        return s[start:start+max_len]

# 最长公共子序列
class Solution:
    def longestsequence(text1, txet2):
        # dp[i][j]表示text1[0:i]和text2[0:j]的最长公共子序列长度
        m, n = len(text1), len(text2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i-1] == text2[j-1]:
                    # 状态转移方程,即text1的前i个字符与text2的前j个字符的LCS长度,等于text1的前i-1个字符与text2的前j-1个字符的LCS长度加1
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]


# 72. 编辑距离
class Solution:
    def minDistance(word1, word2):
        # dp[i][j]表示word1[0:i]转换为word2[0:j]所需的最少操作数,
        # dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost), cost=0 if word1[i-1]==word2[j-1] else 1
        m, n = len(word1), len(word2)
        dp = [[0]*(n+1) for _ in range(m+1)]

        for i in range(1, m+1):
            dp[i][0] = i
        
        for j in range(1, n+1):
            dp[0][j] = j
        
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+1)
        return dp[m][n]



"""75. 颜色分类
给定一个包含红色、白色和蓝色,即元素为 0、 1 和 2 的数组,对它们进行排序,使得相同颜色的元素相邻,并按照 0、1、2 的顺序排列。
输入:nums = [2,0,2,1,1,0]
输出:[0,0,1,1,2,2]
"""

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        count = [0,0 ,0]
        for num in nums:
            count[num] += 1
        index = 0
        # 
        for i in range(3):
            for _ in range(count[i]):
                nums[index] = i
                index += 1


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




# 121.股票最佳买卖问题
# 给定一个数组 prices ，它的第 i 个元素 prices[i] 表示一支给定股票第 i 天的价格。
# 你只能选择 某一天 买入这只股票，并选择在 未来的某一个不同的日子 卖出该股票。设计一个算法来计算你所能获取的最大利润。
# dp[i] = max(dp[i-1], price[i]-minprice)
# 时间复杂度O(n)，空间复杂度O(n)
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minprice = float('inf')
        # dp[i]表示第i天卖出股票的最大利润
        dp = [0] * len(prices)
        for i in range(1, len(prices)):
            minprice = min(minprice, prices[i-1])
            dp[i] = max(dp[i-1], prices[i]-minprice)
        return dp[-1]


# 55. 跳跃游戏
# 给你一个非负整数数组 nums ，你最初位于数组的 第一个下标 。数组中的每个元素代表你在该位置可以跳跃的最大长度。
# 判断你是否能够到达最后一个下标，如果可以，返回 true ；否则，返回 false 。
"""
输入:nums = [2,3,1,1,4]
输出:true
解释:可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
"""
class Solution:
    def canjump(nums):
        right_max = 0
        # 需要看每个位置+自己位置的值能否大于当前位置的索引，从而能否到达下一个位置
        for idx, val  in enumerate(nums[:-1]):
            right_max = max(right_max, idx + val)
            # 如果当前位置已经超过了最远位置,则无法到达最后一个位置
            if idx >= right_max:
                return False
        return True

# 45. 跳跃游戏 II
# 给你一个非负整数数组 nums ,你最初位于数组的第一个下标。数组中的每个元素代表你在该位置可以跳跃的最大长度。
# 你的目标是使用最少的跳跃次数到达数组的最后一个下标。
"""
输入:nums = [2,3,1,1,4]
输出:2
解释:跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 的位置,跳 1 步,然后跳 3 步到达数组的最后一个位置。
"""
class Solution:
    def jump(self, nums: List[int]) -> int:
        jumps = 0
        current_end = 0 # 当前跳跃的最远位置
        farthest = 0
        for i in range(len(nums) - 1):
            # 更新当前跳跃的最远位置
            farthest = max(farthest, i + nums[i])
            # 当到达当前跳跃的最远位置时,需要进行下一次跳跃
            if i == current_end:
                jumps += 1
                current_end = farthest
        return jumps

# 763. 划分字母区间
# 给定一个字符串 s ,将 s 划分为尽可能多的片段,同一字母最多出现在一个片段中。返回一个表示每个字符串片段的长度的列表。
"""
输入:s = "ababcbacadefegdehijhklij"
输出:[9,7,8]
解释:
划分结果为 "ababcbaca", "defegde", "hijhklij" 。
每个字母最多出现在一个片段中。
像是 "ababcbacadefegde", "hijhklij" 这样的划分是不正确的。
"""
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        # 解题思路: 记录每个字母最后出现的位置,然后遍历字符串,更新当前片段的结束位置,当遍历到当前片段的结束位置时,将片段长度加入结果列表
        last = {}
        for i, c in enumerate(s):
            last[c] = i
        
        start = end = 0
        res = []
        for i, c in enumerate(s):
            end = max(end, last[c])
            if i == end:


                res.append(end-start +1)
                start = end + 1
        return res
        
