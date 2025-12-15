##############################################################
# 链表相关题目
##############################################################



"""
160. 相交链表
https://leetcode.cn/problems/intersection-of-two-linked-lists/
编写一个程序，找到两个单链表相交的起始节点。
如果两个链表没有交点，返回 null 。
题目数据 保证 整个链式结构中不存在环。
注意，函数返回结果后，链表必须 保持其原始结构 。
示例 1:
输入:intersectVal = 8, listA = [4,1,8,4,5], listB = [5,0,1,8,4,5], skipA = 2, skipB = 3
输出:Intersected at '8'
解释:相交节点的值为 8 （注意，如果两个链表相交则不能为 0）。从各自的表头开始算起，链表 A 为 [4,1,8,4,5]，链表 B 为 [5,0,1,8,4,5]。在 A 中，相交节点前有 2 个节点；在 B 中，相交节点前有 3 个节点。
示例 2:
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
# 方法：双指针，时间复杂度O(n)，空间复杂度O(1)，n为两个链表的长度
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        """
        :type head1, head2: ListNode
        :rtype: ListNode
        """
        if not headA or not headB: # 如果两个链表为空，则返回None
            return None
        pa = headA # 定义两个指针，分别指向两个链表的头节点
        pb = headB
        while pa != pb:
            pa = pa.next if pa else headB # 如果指针到达链表尾部，则将其指向另一个链表的头节点
            pb = pb.next if pb else headA # 这样可以让两个指针走过相同的距离，最终相遇或抵达交点
        return pa
    

"""
206. 反转链表
https://leetcode.cn/problems/reverse-linked-list/description/
给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。
示例 1:
输入:head = [1,2,3,4,5]
输出:[5,4,3,2,1]"""
# 方法：递归,时间复杂度O(n)，空间复杂度O(n)
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next: # 递归终止条件
            return head
        newHead = self.reverseList(head.next) # 递归调用
        head.next.next = head # 反转当前节点, 将当前节点的下一个节点的next指向当前节点,即反转当前节点,nk+1->nk
        head.next = None # 防止形成环（如2→3和3→2同时存在）
        return newHead

# 方法：迭代,时间复杂度O(n)，空间复杂度O(1)
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        last = None
        while head:
            tmp = head.next # 保存当前节点的下一个节点
            head.next = last # 将当前节点的next指向last
            last = head # 更新last
            head = tmp # 更新head
        return last # 返回最后一个节点

"""
234. 回文链表
https://leetcode.cn/problems/palindrome-linked-list/description/
给你一个单链表的头节点 head ，请你判断该链表是否为回文链表。如果是，返回 true ；否则，返回 false 。
示例 1:
输入:head = [1,2,2,1]
输出:true
示例 2:
输入:head = [1,2]
输出:false"""

# 方法一：将链表的值存入列表，然后判断列表是否为回文，时间复杂度O(n)，空间复杂度O(n)
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        vals = []
        while head:
            vals.append(head.val) # 将链表的值添加到列表vals中
            head = head.next # 移动到下一个节点
        return vals == vals[::-1] # 比较vals和vals的逆序列表是否相等
    

"""
141. 环形链表    https://leetcode.cn/problems/linked-list-cycle/
给定一个链表，判断链表中是否有环。
如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 """
# 方法：哈希表，时间复杂度O(n)，空间复杂度O(n)
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False


"""
1442. 环形链表 II
https://leetcode.cn/problems/linked-list-cycle-ii/description/
给定一个链表，返回链表开始入环的第一个节点。 如果链表无环，则返回 null。
"""
# 方法：哈希表，时间复杂度O(n)，空间复杂度O(n)
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        seen = set()
        while head:
            if head in seen:
                return head # 如果已经访问过这个节点，说明它是环路中的一个节点
            seen.add(head) # 否则，将其加入已访问节点的集合
            head = head.next # 继续访问下一个节点
        return None

"""21. 合并两个有序链表
https://leetcode.cn/problems/merge-two-sorted-lists/
将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。
输入:list1 = [1,2,4], list2 = [1,3,4]
输出:[1,1,2,3,4,4]"""

# 方法：递归，时间复杂度O(m+n)，空间复杂度O(m+n)
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]):
        # 终止条件:当两个链表都为空时，表示我们对链表已合并完成。
        # 如何递归:我们判断 list1 和 list2 头结点哪个更小，然后较小结点的 next 指针指向其余结点的合并结果。（调用递归）
        if not list1: return list2
        if not list2: return list1

        if list1.val <=list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2
        
        # 方法：迭代，时间复杂度O(m+n)，空间复杂度O(1)
        # tail = dummy = ListNode(0)  # 合并两个有序链表
        # while left and right:
        #     if left.val < right.val:
        #         tail.next = left #  tail指向left
        #         left = left.next
        #     else:
        #         tail.next = right #  tail指向right
        #         right = right.next
        #     tail = tail.next
        # tail.next = left or right
        # return dummy.next
        
"""2. 两数相加
https://leetcode.cn/problems/add-two-numbers/
给你两个 非空 的链表，表示两个非负的整数。它们每位数字都是按照 逆序 的方式存储的，并且每个节点只能存储 一位数字。
请你将两个数相加，并以相同形式返回一个表示和的链表。
你可以假设除了数字 0 之外，这两个数都不会以 0 开头。
输入:l1 = [2,4,3], l2 = [5,6,4]
输出:[7,0,8]
解释:342 + 465 = 807."""
# 方法：迭代，时间复杂度O(max(m,n))，空间复杂度O(max(m,n))
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = cur = ListNode(0) # 哑节点,cur指向哑节点
        carry = 0
        while l1 or l2 or carry: # 当l1,l2,carry为空时，结束循环
            if l1: 
                carry += l1.val # 将l1的值加到carry上
                l1 = l1.next # 移动l1指针到下一个节点
            if l2:
                carry += l2.val # 将l2的值加到carry上
                l2 = l2.next # 移动l2指针到下一个节点
            cur.next = ListNode(carry % 10) # 创建新节点，值为carry的个位数
            cur = cur.next # 移动cur指针到下一个节点
            carry //= 10 # 更新carry为carry的十位数
        return dummy.next # 返回哑节点的下一个节点，即结果链表的头节点


"""19. 删除链表的倒数第 N 个 节点
https://leetcode.cn/problems/remove-nth-node-from-end-of-list/
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。
输入:head = [1,2,3,4,5], n = 2
输出:[1,2,3,5]w
"""
# 方法：双指针，时间复杂度O(n)，空间复杂度O(1)
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        first = second = dummy = ListNode(0) # 哑节点 first second,双指针
        dummy.next = head # 哑节点指向head
        for _ in range(n + 1):
            first = first.next # 第一个指针先移动n+1步
        while first:
            first = first.next # 然后两个指针同时移动，直到第一个指针到达链表末尾,即两个指针相差n步
            second = second.next
        second.next = second.next.next # 删除倒数第n个节点
        return dummy.next # 返回哑节点的下一个节点，即结果链表的头节点
    

"""24. 两两交换链表中的节点
https://leetcode.cn/problems/swap-nodes-in-pairs/
给你一个链表，两两交换其中相邻的节点，并返回交换后的链表。你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。
输入:head = [1,2,3,4]
输出:[2,1,4,3]
"""
# 方法一：迭代
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0) # 哑节点
        dummy.next = head
        prev = dummy
        while head and head.next:
            first = head # 当前节点  1
            second = head.next # 下一个节点  2

            # 交换节点
            prev.next = second # 前一个节点的下一个节点指向第二个节点  0->2
            first.next = second.next # 第一个节点的下一个节点指向第二个节点的下一个节点 1->3
            second.next = first # 第二个节点的下一个节点指向第一个节点 2->1

            # 更新指针
            prev = first # 前一个节点指向第一个节点 0->1
            head = first.next # head指向下一个节点 1->3
        return dummy.next # 返回新链表的头节点  0->2->1->3
    
# 方法二： 
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        while pre.next and pre.next.next:
            a = pre.next
            b = a.next
            pre.next, b.next, a.next = b, a, b.next
            pre = a
        return dummy.next
"""25. K 个一组翻转链表
https://leetcode.cn/problems/reverse-nodes-in-k-group/
给你一个链表，每 k 个节点一组进行翻转，请你返回翻转后的链表。k 是一个正整数，它的值小于或等于链表的长度。
如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
输入:head = [1,2,3,4,5], k = 2
输出:[2,1,4,3,5]
"""
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy

        while True:
            tail = prev
            for i in range(k):
                tail = tail.next # 找到子链表的尾节点
                if not tail: # 如果子链表长度不足k，则直接返回
                    return dummy.next # 返回新链表的头节点
            
            next_node = tail.next # 保存子链表的下一个节点
            # 反转子链表
            prev_next, curr = prev.next, prev.next.next # prev_next指向子链表的头节点，curr指向子链表的第二个节点
            for i in range(k - 1):
                tmp = curr.next # 保存当前节点的下一个节点
                curr.next = prev.next # 将当前节点的next指向prev的下一个节点
                prev.next = curr # 将prev的下一个节点指向当前节点
                curr = tmp # 更新curr指针到下一个节点
                prev_next.next = next_node # 将子链表的头节点的next指向下一个节点

            # 更新指针
            prev = prev_next
            tail = prev
            if not next_node: # 如果下一个节点为空，则结束循环
                break

        return dummy.next # 返回新链表的头节点

# 方法二：将链表的节点值依次添加，然后k个一组反转链表,返回新链表的头节点
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        res = []
        while head:
            res.append(head.val) # 将链表的值添加到列表res中
            head = head.next
        num = len(res) // k # 计算可以完整反转的子链表数量
        num_res = len(res) % k # 计算剩余节点数量

        tmp = []
        for i in range(num):
            s = res[k*i:k*(i+1)][::-1] # 反转子链表
            tmp += s # 将反转后的子链表添加到tmp中
        if num_res == 0: # 如果剩余节点数量为0，则直接返回tmp
            tmp = tmp 
        else:
            tmp = tmp + res[-num_res:]

        # 创建新链表
        head = root = ListNode(0)
        for i in tmp:
            node = ListNode(i) # 创建新节点
            head.next = node # 将新节点添加到链表中
            head = head.next # 移动head指针到下一个节点
        return root.next


class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        res =[]
        while head:
            res.append(head.val)
            head = head.next
        dummy = cur = ListNode(0)
        for i in range(0, len(res), k):
            group = res[i:i+k]
            if len(group) == k:
                group.reverse()
            for val in group:
                cur.next = ListNode(val)
                cur = cur.next
        return dummy.next 

"""148. 排序链表
https://leetcode.cn/problems/sort-list/
给你链表的头结点 head ，请你返回 升序排列 的结果链表。
"""
# 方法一：排序，时间复杂度O(nlogn)
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        vals = []
        while head:
            vals.append(head.val)
            head = head.next
        vals.sort()

        root = dummy = ListNode(0)
        for i in vals:
            dummy.next = ListNode(i)
            dummy = dummy.next
        return root.next

# 方法二：归并排序
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next: # 如果链表为空或者只有一个节点，直接返回ws
            return head
        
        # 使用快慢指针找到链表的中点
        slow, fast = head, head.next
        while fast and fast.next:
            slow = slow.next # 慢指针每次移动一个节点
            fast = fast.next.next # 快指针每次移动两个节点
        
        mid = slow.next # 中点
        slow.next = None
        
        left = self.sortList(head)  # 递归排序左半部分
        right = self.sortList(mid)  # 递归排序右半部分

        tail = dummy = ListNode(0)  # 合并两个有序链表
        while left and right:
            if left.val < right.val:
                tail.next = left #  tail指向left
                left = left.next
            else:
                tail.next = right #  tail指向right
                right = right.next
            tail = tail.next
        tail.next = left or right
        return dummy.next

"""23. 合并K个升序链表
https://leetcode.cn/problems/merge-k-sorted-lists/
给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
"""  

# 方法一：两两合并
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists or len(lists) == 0:
            return None
        while len(lists) > 1:
            merged = self.mergeTwoLists(lists[0], lists[1]) # 合并两个链表
            lists = [merged] + lists[2:]
        return lists[0]

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1: return l2
        if not l2: return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2 
    
# 方法二：分治法
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        if not lists:return 
        n = len(lists)
        return self.merge(lists, 0, n-1)
    def merge(self,lists, left, right):
        if left == right:
            return lists[left] # 只有一个元素时直接返回
        mid = left + (right - left) // 2 # 中间位置
        l1 = self.merge(lists, left, mid) # 递归合并左半部分
        l2 = self.merge(lists, mid+1, right) # 递归合并右半部分
        return self.mergeTwoLists(l1, l2) # 合并左右两部分
    def mergeTwoLists(self,l1, l2):
        if not l1:return l2
        if not l2:return l1
        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

# 方法三: 直接将节点值存入列表，然后排序，最后再返回节点
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # 把节点值存入列表，然后排序，最后再返回节点
        vals = []
        for list in lists:
            while list:
                vals.append(list.val)
                list = list.next
        vals.sort() # 排序所有节点值
        root = dummy = ListNode(0)
        for i in vals:
            dummy.next = ListNode(i)
            dummy = dummy.next
        return root.next
    

"""
146. LRU缓存机制
https://leetcode.cn/problems/lru-cache/
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束 的数据结构。
实现 LRUCache 类：  LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存。  int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。 
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。当缓存容量达到 capacity 时，则应该 逐出 最久未使用的键值对，以腾出空间进行插入。
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
"""

# 方法：使用字典和列表实现LRU缓存机制
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {} # 初始化一个空字典
        self.order = [] # 初始化一个空列表, 用于记录最近使用的key

    def get(self, key: int) -> int:
        if key in self.cache:
            self.order.remove(key) # 将key从最近使用的位置移除
            self.order.append(key) # 将key添加到最近使用的位置
            return self.cache[key] # 返回key对应的value
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache: # 如果key已经存在，则更新value，并将key移动到最近使用的位置
            self.cache[key] = value
            self.order.remove(key) 
        else: # 否则，如果key不存在，则添加新的key-value对，并将key移动到最近使用的位置
            if len(self.cache) >= self.capacity:
                oldest = self.order.pop(0) # 移除最久未使用的key
                del self.cache[oldest] # 删除最久未使用的key对应的value
            self.cache[key] = value # 添加新的key-value对
        self.order.append(key) # 将key添加到最近使用的位置

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)csrrs[]