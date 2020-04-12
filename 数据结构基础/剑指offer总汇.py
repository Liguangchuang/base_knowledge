#########################################################
##二叉树中和为某一值的路径 
##输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
##路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径

##这道题要对递归的内部很了解才能解答出来！！！
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def FindPath(self, root, expectNumber):
        self.one_path = []
        self.all_path = []
        
        self.get_all_path(root, expectNumber)
        
        return self.all_path
        
    def get_all_path(self, root, expectNumber):
        if root is None:
            return 
        self.one_path.append(root.val)
        self.get_all_path(root.left, expectNumber)
        self.get_all_path(root.right, expectNumber)
        
        is_leaf = (root.left is None and root.right is None)
        if is_leaf:  ##仿真回退的时候，把那些路径也算上去
            if sum(self.one_path[:]) == expectNumber:
                self.all_path.append(self.one_path[:])  ##[:]这个是一定要加的，记住了！！
        self.one_path.pop()
        
#########################################################
##二叉树的下一个节点
##给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，
##树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
#有一点一定要注意，python的抽象中，“根节点”等价于“树”
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    def GetNext(self, pNode):
        if pNode is None:
            return None
        
        pRoot = pNode
        while pRoot.next:
            pRoot = pRoot.next
            
        self.inOrderTra_array = []
        self.inOrder_Traverse(pRoot)
        
        if self.inOrderTra_array.index(pNode) == len(self.inOrderTra_array) - 1:
            return None
        else:
            return self.inOrderTra_array[self.inOrderTra_array.index(pNode) + 1]

    def inOrder_Traverse(self, pRoot):
        if pRoot is None:
            return None
        self.inOrder_Traverse(pRoot.left)
        self.inOrderTra_array.append(pRoot)
        self.inOrder_Traverse(pRoot.right)
        
        
#########################################################
##输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。
##假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
##例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def reConstructBinaryTree(self, pre, tin):
        if pre ==[] or tin == []:
            return None
        
        root_value = pre[0]
        tin_root_index = tin.index(root_value)

        #对每棵树，依次创建根节点，左子节点，右子节点
        pRoot = TreeNode(root_value)
        pRoot.left = self.reConstructBinaryTree(pre[1: tin_root_index+1], tin[0: tin_root_index])
        pRoot.right = self.reConstructBinaryTree(pre[tin_root_index+1: ], tin[tin_root_index+1: ])
        return pRoot

        
#########################################################
###把二叉树打印成多行 
##从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。

##一定要深刻了解python的引用机制，“=”在list，df，链表...（结构变量）时不是赋值，而是引用，非常重要！！！
##标准库是可以直接import的，是完全能允许的
##在用pandas做特征的时候，最好使用df_ = df.copy()；否则很容易出错！！！！
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from copy import copy
class Solution:
    # 返回二维列表[[1,2],[4,5]]
    def Print(self, pRoot):
        if pRoot is None: 
            return []

        #初始化
        result = [[pRoot.val]]
        qu_root = []
        qu_sub = []
        qu_root.append(pRoot)
        
        while qu_root:
            bT = qu_root.pop(0)
            if bT.left:
                qu_sub.append(bT.left)
            if bT.right:
                qu_sub.append(bT.right)
                
            if qu_root == []:
                qu_root = copy(qu_sub)
                
                res = []
                while qu_sub:
                    res.append(qu_sub.pop(0).val)
                if res:
                    result.append(res)
                    
        return result

        
#########################################################
##二叉搜索树的后序遍历序列 
##输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。
##如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。

##根据后序遍历的规律：序列中的最后一个数字是树的根节点 ，
##根据搜索二叉树的规律：数组中前面的数字可以分为两部分；第一部分是左子树节点的值，都比根节点的值小；第二部分是右子树节点的值，都比根节点的值大，
##后面用递归分别判断前后两部分是否符合以上原则
class Solution:
    def VerifySquenceOfBST(self, sequence):
        if len(sequence) == 0:
            return False
        
        length = len(sequence)
        root_value = sequence[length - 1]
        
        # 找出左子节点的临界点：二叉搜索树中，左子树节点小于根节点
        for i in range(length):
            if sequence[i] > root_value:
                break
                
        # 在右节点中，看是否符合搜索树的要求；
        # 二叉搜索树中，右子树的节点都大于根节点
        for j in range(i, length):
            if sequence[j] < root_value:
                return False
                
        # 判断左子树是否为二叉树
        left = True
        if i > 0:  # >0 才有左子节点，否则直接判断为True
            left = self.VerifySquenceOfBST(sequence[0: i])
            
        # 判断右子树是否为二叉树
        right = True
        if i < length-1:  #<length-1 才有右子树，否则直接判断为True
            right = self.VerifySquenceOfBST(sequence[i: length-1])  #不包括根节点
        
        return left and right
    
    
#########################################################
###二叉树的深度 
##输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，
##最长路径的长度为树的深度
class Solution:
    def TreeDepth(self, pRoot):
        if pRoot is None:
            return 0
        depth_left = self.TreeDepth(pRoot.left)
        depth_right = self.TreeDepth(pRoot.right)
 		return max(depth_left, depth_right) + 1   
            
            
#########################################################
##判断一棵树是否为平衡二叉树 
##输入一棵二叉树，判断该二叉树是否是平衡二叉树。

##平衡二叉树：对于任意节点，1、左子树与右子树都是平衡二叉树
##							2、左子树与右子树的高度差的绝对值相差1；典型的递归结构
##做递归的题目，切忌研究底层细节；只要把握“上层的整体实现”和“退出条件”就可以了，细节交给计算机处理
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

##最基础的做法：计算树深的时候会出现多次遍历，效率比较低
class Solution:
    def IsBalanced_Solution(self, pRoot):
        if pRoot is None:
            return True
        depth_left = self.get_TreeDepth(pRoot.left)
        depth_right = self.get_TreeDepth(pRoot.right)
        if abs(depth_left - depth_right) > 1:
            return False
        else:
            return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
        
    def get_TreeDepth(self, pRoot):
        if pRoot is None:
            return 0
        depth_left = self.get_TreeDepth(pRoot.left)
        depth_right = self.get_TreeDepth(pRoot.right)
        return max(depth_left, depth_right) + 1
    
    

#########################################################
###二叉树的镜像
'''
操作给定的二叉树，将其变换为源二叉树的镜像。

输入描述:
二叉树的镜像定义：源二叉树 
            8
           /  \
          6   10
         / \  / \
        5  7 9 11
        镜像二叉树
            8
           /  \
          10   6
         / \  / \
        11 9 7  5
'''

###将所有的左右子节点交换！！！
class Solution:
    # 返回镜像树的根节点        
    def Mirror(self, root):
        if root is None:
            return 
            
        root.right, root.left = root.left, root.right  #交换左右子节点
        
        #递归，对于每个子树，也交换左右子节点！！
        self.Mirror(root.left)
        self.Mirror(root.right)
        
        

#########################################################
###二叉搜索树的第k小结点 
##给定一棵二叉搜索树，请找出其中的第k小的结点。
##例如，（5，3，7，2，4，6，8）中，按结点数值大小顺序第三小结点的值为4。

##在类内使用变量，可以在任意一个函数内加“self.”，如self.res = []，则self.res在类的全局内都可以使用
##类内全部变量的使用，要加“self.”
##调用类内函数，要加“self.”，包括递归调用；如：self.inOrder_Traverse(pRoot.left)
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回对应节点TreeNode
    def KthNode(self, pRoot, k):
        self.inOrderTra_array = []
        self.inOrder_Traverse(pRoot)
    
        if k < 1 or k > len(self.inOrderTra_array):
            return None
        else:
            return self.inOrderTra_array[k-1]
            
    def inOrder_Traverse(self, pRoot):
        if pRoot is None:
            return 
        self.inOrder_Traverse(pRoot.left)
        self.inOrderTra_array.append(pRoot)
        self.inOrder_Traverse(pRoot.right)
    
    

#########################################################
## 从上往下打印二叉树
##从上往下打印出二叉树的每个节点，同层节点从左至右打印。

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        tmp = []
        
        qu = []
        qu.append(root)
        
        while qu:
            bT = qu.pop(0)
            if bT is None:
                continue
            qu.append(bT.left)
            qu.append(bT.right)
            tmp.append(bT.val)

        return tmp
        

#########################################################
###对称的二叉树 
##请实现一个函数，用来判断一颗二叉树是不是对称的。
##注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isSymmetrical(self, pRoot):
        if pRoot is None:
            return True
        else:
            return self.is_subTree_symmetry(pRoot.left, pRoot.right)
    
    def is_subTree_symmetry(self, p1, p2):
        if p1 == None and p2 == None:  
            return True  #最终的判断
        elif (p1 != None and p2 != None) and (p1.val == p2.val):
            return self.is_subTree_symmetry(p1.left, p2.right) and self.is_subTree_symmetry(p1.right, p2.left)
        else: 
            return False  #最终的判断
            
            
            
###########################################
##从尾到头打印链表
##输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        if listNode is None:
            return print_list
            
        print_list = []
        
        while listNode:
            print_list.append(listNode.val)
            listNode = listNode.next
            
        print_list.reverse()
        
        return print_list

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        if listNode is None:
            return print_list
            
        print_list = []
        stack = []  #先进栈，后出栈，结果自然就倒过来了！！
        
        while listNode:
            stack.append(listNode.val)
            listNode = listNode.next
            
        while stack:
            print_list.append(stack.pop())
        
        return print_list			
        
#########################################################
##链表中倒数第k个结点 
##输入一个链表，输出该链表中倒数第k个结点。

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
##最好的解法：
##设置两个指针，第一个指针先走k步，然后两个指针同时走，
##第一个指针结束时，第二个指针恰好走到倒数第k个点
class Solution:
    def FindKthToTail(self, head, k):
        firstNode = head
        secondNonde = head

        step = 0
        while firstNode:
            firstNode = firstNode.next
            if step >= k:
                secondNonde = secondNonde.next
            step += 1
        if k > step or k <= 0:
            return None
        return secondNonde
        

##两次遍历：先记录一共有num个节点，求倒数k个节点，等价于删除前num-k和节点
class Solution:
    def FindKthToTail(self, head, k):
        pNode = head
        n_Node = 0
        
        while pNode:
            pNode = pNode.next
            n_Node += 1
        
        if k > n_Node or k <= 0:
            return None
        
        pNode2 = head
        k_th = n_Node - k
        for _ in range(k_th):
            pNode2 = pNode2.next
        return pNode2
    
###python的解法
class Solution:
    def FindKthToTail(self, head, k):
        pNode = head
        tmp = []
        
        while pNode:
            tmp.append(pNode)
            pNode = pNode.next
        
        if k > len(tmp) or k <= 0:
            return None
        return tmp[-k]


#########################################################
##反转链表
##输入一个链表，反转链表后，输出新链表的表头。
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def ReverseList(self, pHead):
        if pHead is None or pHead.next is None:
            return pHead
            
        pNode = pHead
        pLast = None
        while pNode:
            pTemp = pNode.next  #先把后续节点缓存下来
            pNode.next = pLast  #第一个节点连到最后节点；并且原来的链接自动断开
            pLast = pNode  #更新最后节点
            pNode = pTemp  #更新首节点
        return pLast
            
        
    

#########################################################	
###链表中环的入口结点 
##给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。


###点子简直不要太绝妙！！！
'''
图片：https://uploadfiles.nowcoder.net/images/20170422/943729_1492841744777_3BB680C9CBA20442ED66C5066E1F7175
思路：
1、设置两个指针fast和slow，fast以slow两倍速度前进，
2、如果没有环，那么fast和slow不会相遇此时返回None；
3、如果有环，那fast和slow肯定会再次相遇; 
   相遇的时候，fast刚好比slow多走了一圈环的长度；fast走过的距离为a + b + c + b，而slow走过的距离为a + b，
   因为fast是slow速度的两倍，则有a+b+c+b = 2*(a+b)，解出a=c;
   因此，设置第三个指针p，从X处，以和slow指针相同的速度前进，当它两相遇时，即为环的起点Y处！
'''
class Solution:
    def EntryNodeOfLoop(self, pHead):
        if pHead ==None or pHead.next == None:
            return None
        
        pFast = pHead
        pSlow = pHead
        while pFast.next.next:
            pFast = pFast.next.next  ##while那里的“两个next”判断主要是为了这个
            pSlow = pSlow.next
            if pFast == pSlow:
                newNode = pHead
                while newNode != pSlow:
                    newNode = newNode.next
                    pSlow = pSlow.next
                return newNode
        return None
                
    
    

#########################################################	
##两个链表的第一个公共节点
##输入两个链表，找出它们的第一个公共结点。

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        if pHead1 is None or pHead2 is None:
            return None
        
        #分别计算两个链表的长度
        p1, p2 = pHead1, pHead2
        length1 = length2 = 0
        while p1:
            p1 = p1.next
            length1 += 1
        while p2:
            p2 = p2.next
            length2 += 1
        
        #先让长的走(长-短)步，直到两个链表的长度一致
        p1, p2 = pHead1, pHead2
        if length1 > length2:
            while length1 > length2:
                p1 = p1.next
                length1 -= 1
        else:
            while length2 > length1:
                p2 = p2.next
                length2 -= 1
        
        #当两个链表一致的时，同时走直到发现相同点
        while p1 != p2:
            p1 = p1.next
            p2 = p2.next
        
        return p1
                
    

#########################################################	
####合并两个排序的链表 
##输入两个单调递增的链表，输出两个链表合成后的链表，
##当然我们需要合成后的链表满足单调不减规则。
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        #递归的结束条件，递归一定要从大的方面来思考
        if pHead1 is None:
            return pHead2
        elif pHead2 is None:
            return pHead1
            
        pMerge = None
        if pHead1.val < pHead2.val:
            pMerge = pHead1
            pMerge.next = self.Merge(pHead1.next, pHead2)
        else:
            pMerge = pHead2
            pMerge.next = self.Merge(pHead1, pHead2.next)
            
        return pMerge

        

#########################################################
##复杂链表的复制 
##输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，
##另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head

'''
Python相对于C有一个非常大差别的地方，变量的是“引用”机制
这种表示一段区域的变量，比如list，链表等，赋值（比如a_list = b_list）表示同一个区域用了两个名字表示，
当对一个变量进行操作时，两个变量的值都会改变（比如a_list原来是[1,2,3]，a_list.append(2)；这时候a_list和b_list同时变为[1,2,3,2]）

两个链表，pNode = pHead；然后对pNode操作，最后的结果pNode = None，但是pHead变成了更长的2倍复制链表
因为头节点处的“区域”是一样的，你不断地通过操作pNode，改变“这些区域的链接关系”，最后从pHead的“初始区域”出发，当然会得到一条新的链表
这不是Python的引用机制，是“链接关系”的改变，就是头部区域不变，但链接关系变了！！！
'''
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        if pHead is None:
            return None

        pDoubleHead = pHead

        pDoubleHead = self.cloneNodes(pDoubleHead)
        pDoubleHead = self.connectRandomNodes(pDoubleHead)
        pCloneHead = self.reConnetNode(pDoubleHead)

        return pCloneHead
        
    #复制节点，并向前链接起来
    def cloneNodes(self, pDoubleHead):
        pNode = pDoubleHead
        while pNode:
            pCloneNode = RandomListNode(pNode.label)
            pCloneNode.next = pNode.next  #先后顺序不可颠倒
            pNode.next = pCloneNode
            pNode = pNode.next.next
        return pDoubleHead
        
    #克隆节点的random指针链接
    def connectRandomNodes(self, pDoubleHead):
        pNode = pDoubleHead
        while pNode:
            if pNode.random:
                pNode.next.random = pNode.random.next
            pNode = pNode.next.next
        return pDoubleHead
        
    #双倍链表 断开为  原链表 和 复制链表
    def reConnetNode(self, pDoubleHead):
        pCloneHead = pCloneNode = pDoubleHead.next

        while pCloneNode.next:
            pDoubleHead.next = pCloneNode.next
            pDoubleHead = pCloneNode.next
            pCloneNode.next = pDoubleHead.next
            pCloneNode = pDoubleHead.next
        pDoubleHead.next = None

        return pCloneHead


#########################################################
##用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型
class Solution:
    def __init__(self):
        self.stack_1 = []
        self.stack_2 = []

    def push(self, node):
        self.stack_1.append(node)
        
    def pop(self):
        if self.stack_2 == []:
            while self.stack_1 != []:
                self.stack_2.append(self.stack_1.pop())
        return self.stack_2.pop()

        

#########################################################	
##包含min函数的栈 
##定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数
##（时间复杂度应为O（1））。
class Solution:
    def __init__(self):
        self.stack = []
        self.top_min_stack = []
        
    def push(self, node):
        self.stack.append(node)
        
        if (self.top_min_stack == []) or (node < self.top_min_stack[-1]):  ##真的是无语了，这里调换顺序就报错
            self.top_min_stack.append(node)
        else:
            self.top_min_stack.append(self.top_min_stack[-1])

    def pop(self):
        self.top_min_stack.pop()
        return self.stack.pop()
        
    def top(self):
        return self.stack[-1]

    def min(self):
        return self.top_min_stack[-1]
    
    

#########################################################	
##栈的压入、弹出序列 
##输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。
##假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，
##但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

class Solution:
    def IsPopOrder(self, pushV, popV):
        stack = []
        while True:
            if pushV != []:
                stack.append(pushV.pop(0))
                
            print(pushV)
            print(popV)

            while (stack != []) and (stack[-1] == popV[0]):  ##草拟大爷，这个调换顺序就会报错，
                stack.pop()                                  ##一定要记住，他是按顺序判断的，
                popV.pop(0)                                  ##先判断stack != []；再判断stack[-1] == popV[0]

            if stack == []:
                return True

            if (pushV == []) and (stack != []) and (stack[-1] != popV[0]):
                return False

                
###########################################
##顺时针打印矩阵 

##打印第一行，然后删除第一行，然后对矩阵旋转。
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        print_list = []
        while matrix:
            for c in range(0, len(matrix[0])):
                print_list.append(matrix[0][c])
            matrix = self.transpose(matrix)
        return print_list

    def transpose(self, matrix):
        del matrix[0]
        if matrix == []:
            return

        new_matrix = []
        for c in range(len(matrix[0])-1, -1, -1):
            new_row = []
            for r in range(0, len(matrix)):
                new_row.append(matrix[r][c])

            new_matrix.append(new_row)
        return new_matrix

        
##一圈一圈打印，
class Solution:
    # matrix类型为二维列表，需要返回列表
    def printMatrix(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0])

        print_list = []
        start = 0

        while (cols > start * 2) and (rows > start * 2):  #打印的结束条件
            circle_list = self.print_mat_cricle(matrix, start)
            print_list += circle_list

            start += 1

        return print_list

    def print_mat_cricle(self, matrix, start):
        circle_list = []

        end_x = len(matrix[0])-start-1
        end_y = len(matrix)-start-1

        #从左往右打印一行
        for c in range(start, end_x+1):
            circle_list.append(matrix[start][c])

        #从上往下打印一列
        if start < end_y:
            for r in range(start+1, end_y+1):
                circle_list.append(matrix[r][end_x])

        #从右往左打印一行
        if (start < end_y) and (start < end_x):
            for c in range(end_x-1, start-1, -1):
                circle_list.append(matrix[end_y][c])

        #从下往上打印一列
        if (start < end_y-1) and (start < end_x):
            for r in range(end_y-1, start, -1):
                circle_list.append(matrix[r][start])

        return circle_list
        
        
###########################################
##替换空格
##请实现一个函数，将一个字符串中的每个空格替换成“%20”。
##例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。

##在一个新的字符串上操作，时间O(n),空间O(n+m)
class Solution:
    def replaceSpace(self, s):
        new_s = ''
        for c in s:
            if c == ' ':
                c = '%20'
            new_s += c
        return new_s


##剑指offer的解法；在原来的字符串上操作，时间O(n),空间O(m)
class Solution:
    def replaceSpace(self, s):
        #统计空格的数量
        l_idx = len(s) - 1
        n_space = 0
        for c in s:
            if c == ' ':
                n_space += 1

        #扩充字符串
        for _ in range(n_space):
            s += '  '  #每个空格增加2个位置
        
        #在扩充字符串上填充
        s = list(s)    #python的字符串是不可变的，需要先转化为list，才能操作
        r_idx = len(s) - 1 
        while l_idx >= 0:
            print(s[l_idx])
            if s[l_idx] == ' ':
                s[r_idx] = '0'
                s[r_idx-1] = '2'
                s[r_idx-2] = '%'
                r_idx -= 3
            else:
                s[r_idx] = s[l_idx]
                r_idx -= 1
            l_idx -= 1
        s = ''.join(s)
        return s
            
            
        
        

        
        
##了解一下python的特性吧，实际不能这样做
class Solution:
    # s 源字符串
    def replaceSpace(self, s):
        return s.replace(' ', '%20')
        
        
        
###########################################
##字符流中第一个不重复的字符
##请实现一个函数用来找出字符流中第一个只出现一次的字符。
##例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。
##当从该字符流中读出前六个字符“"google"时，第一个只出现一次的字符是"l"。
##如果当前字符流没有存在出现一次的字符，返回#字符。

##实际上，这是作弊的做法，因为你使用了内置的哈希
class Solution:
    # 返回对应char
    def __init__(self):
        self.s = ''
        
    def FirstAppearingOnce(self):
        for i in range(len(self.s)):
            if self.s[i] not in self.s[0: i]+ self.s[i+1: ]:
                return self.s[i]
        return '#'
        
    def Insert(self, char):
        self.s += char


##哈希表实现，哈希这个名词不难的!!
class Solution:
    def __init__(self):
        self.hashTable = [0] * 256
        self.s = ""
    
    def FirstAppearingOnce(self):
        for i in range(0, len(self.s)):
            hash_idx = ord(self.s[i])
            if self.hashTable[hash_idx] == 1:
                return self.s[i]
        return '#'
        
    def Insert(self, char):
        self.hashTable[ord(char)] += 1
        self.s += char
        
        
#########################################
##第一个只出现一次的字符，的位置
##在一个字符串中找到第一个只出现一次的字符,并返回它的位置, 
##如果没有则返回 -1
class Solution:
    def FirstNotRepeatingChar(self, s):
        for i in range(len(s)):
            if s[i] not in s[0: i] + s[i+1 :]:
                return i
        return -1 
    

##哈希表实现，其实好简单啊!!!
class Solution:
    def FirstNotRepeatingChar(self, s):
        #构建哈希表
        hashTable = [0] * 256
        for c in s:
            hashTable[ord(c)] += 1  ##根据acci码构建哈希表
            
        for i in range(0, len(s)):
            hash_idx = ord(s[i])
            if hashTable[hash_idx] == 1:
                return i
        return -1 


#########################################        
##左旋转字符串
##对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。
##例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。
###这不是很他妈简单吗?就是把前n位的字符串移动到末尾

class Solution:
    def LeftRotateString(self, s, n):
        s_left = s[0: n]
        s_right = s[n: ]
        return s_right + s_left
        

#########################################
##翻转单词顺序
##student. a am I” --->  “I am a student.”。

##直接用python内置的3个方法；空间O(n)
class Solution:
    def ReverseSentence(self, s):
        word_list = s.split(' ')
        word_list.reverse()  #直接改变原来的list
        
        new_s = ' '.join(word_list)
        return new_s

##reverse函数用栈实现
class Solution:
    def ReverseSentence(self, s):
        stack = s.split(' ')
        word_list = self.reverse(stack)
            
        new_s = ' '.join(word_list)
        return new_s
    
    def reverse(self, stack):
        stack_reverse = []
        while stack:
            stack_reverse.append(stack.pop())
        return stack_reverse

##split函数用左右指针扫描实现，reverse函数用交换实现；
##join函数用 + 连接实现；空间O(n)
##已经比较精髓了!!
class Solution:
    def ReverseSentence(self, s):   
        word_list = self.split(s, ' ')
        word_list = self.reverse(word_list)
        new_s = self.join(word_list, ' ')
        return new_s
        
    def split(self, s, c):
        word_list = []
        s += c
        
        l_idx = 0
        r_idx = 0
        for _ in range(len(s)):
            if s[r_idx] == c:
                word_list.append(s[l_idx: r_idx])
                l_idx = r_idx+1    
            r_idx += 1
        return word_list
        
    def reverse(self, word_list):
        l_idx = 0
        r_idx = len(word_list) - 1
        while l_idx <= r_idx:
            word_list[l_idx], word_list[r_idx] = word_list[r_idx], word_list[l_idx]
            l_idx += 1
            r_idx -= 1
        return word_list
    
    def join(self, word_list, c):
        new_s = ''
        for sub_s in word_list:
            new_s += (sub_s + c)
        new_s = new_s[0: len(new_s)-1]
        return new_s

        
##剑指的标准做法，在原来的字符串上操作，空间O(1)；有点难度
##先翻转字母，再翻转单词
class Solution:
    def ReverseSentence(self, s):
        #先对整体的字符翻转
        s = list(s)  #python的字符串是不可变的，需要先转化为list，才能操作
        l_idx = 0
        r_idx = len(s) - 1
        while l_idx <= r_idx:
            s[l_idx], s[r_idx] = s[r_idx], s[l_idx]
            l_idx += 1
            r_idx -= 1
        
        #再对每个单词的内部字符翻转
        l_idx = 0
        r_idx = 0
        s += ' '
        for _ in range(len(s)):
            if s[r_idx] == ' ':
                l_idx_tmp = l_idx
                r_idx_tmp = r_idx - 1
                while l_idx_tmp <= r_idx_tmp:
                    s[l_idx_tmp], s[r_idx_tmp] = s[r_idx_tmp], s[l_idx_tmp]
                    l_idx_tmp += 1
                    r_idx_tmp -= 1
                l_idx = r_idx+1
            r_idx += 1
        s = s[0: len(s)-1]    
        s = ''.join(s)
        
        return s


###################################################
##二维数组中的查找 
##在一个二维数组中（每个一维数组的长度相同），
##每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
##请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
class Solution:
    # array 二维列表
    def Find(self, target, array):
        if target is None or array is None:
            return None 
        
        row_idx = 0
        col_idx = len(array[0]) - 1
        while (row_idx <= len(array) - 1) and (col_idx >= 0):
            if array[row_idx][col_idx] == target:
                return True
            elif array[row_idx][col_idx] < target:
                row_idx += 1
            elif array[row_idx][col_idx] > target:
                col_idx -= 1
        return False

###################################################
##旋转数组的最小数字 
##把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 
##输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 
##例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 
##NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

##直接二分法咯：
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        if len(rotateArray) == 0:
            return 0
        if rotateArray[0]  == rotateArray[-1]:
            rotateArray[0]

        l_idx = 0
        r_idx = len(rotateArray) - 1
        while l_idx <= r_idx:
            m_idx = (l_idx + r_idx) // 2

            if rotateArray[l_idx] <= rotateArray[m_idx]:
                l_idx = m_idx
            elif rotateArray[r_idx] >= rotateArray[m_idx]:
                r_idx = m_idx
            if r_idx - l_idx == 1:
                break
        return rotateArray[r_idx]
    

#################################################
##调整数组顺序使奇数位于偶数前面 
##输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
##使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
##并保证奇数和奇数，偶数和偶数之间的相对位置不变

##用两个数组分别存奇数和偶数，在链接起来
class Solution:
    def reOrderArray(self, array):
        ji_array = []
        ou_array = []
        
        for elem in array:
            if elem % 2 == 0:
                ou_array.append(elem)
            else:
                ji_array.append(elem)
        new_array = ji_array + ou_array
        return new_array

        
###斐波那数列#####################################
##f(n) = 0				if n = 0
## 	   = 1				if n = 1
##	   = f(n-1) + f(n-2)  if n > 1

##非递归
class Solution:
    def Fibonacci(self, n):
        if n <= 1:
            return n
        
        pre_2 = 0
        pre_1 = 1
        for _ in range(2, n+1):  
            now = pre_1 + pre_2
            pre_2 = pre_1         
            pre_1 = now

        return now 
        
        
##递归（复杂度太高无法运行）
##递归一定要从最顶层开始思考问题；
##假设f(n-1)和f(n-2)已知，求f(n)
##关注 “边界条件” 和 “最后一层的推导” 就可以了，实现的细节交给计算机
class Solution:
    def Fibonacci(self, n):
        if n <= 1:
            return n
            
        #假设f(n-1)和f(n-2)已知，求f(n)
        return self.Fibonacci(n-1) + self.Fibonacci(n-2)  

###2阶跳台阶################################################
##一只青蛙一次可以跳上1级台阶，也可以跳上2级。
##求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

##跳第1级台阶，只有1种可能
##跳第2级台阶，有两种可能（1级1级跳；直接2级）
##...
##跳第n级台阶，上一次从n-1级跳的，那就看跳第n-1级台阶有多少种可能
##		       上一次从n-2级跳的，那就看跳第n-2级台阶有多少种可能
##因此一共的可能是：跳到n-1级的所有可能 + 跳到n-2级的所有可能
##f(n) = 1                if n = 1
##     = 2                if n = 2
##     = f(n-1) + f(n-2)  if n > 2
           
           
class Solution:
    def jumpFloor(self, n):
        if n <= 2:
            return n

        pre_2 = 1
        pre_1 = 2
        for _ in range(3, n+1):
            now = pre_1 + pre_2
            pre_2 = pre_1
            pre_1 = now

        return now 

    
###n阶跳台阶#####################################
##一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。
##求该青蛙跳上一个n级的台阶总共有多少种跳法。

##f(n) = f(n-1) + f(n-2) +... + f(1)      #上一次从n-1级跳，上一次从n级跳，...，上一次从1级跳
##f(n-1) = f(n-2) + f(n-3) +... + f(1)
##一式-二式为：f(n)=2*f(n-1)
class Solution:
    def jumpFloorII(self, n):
        if n == 1:
            return 1

        pre_1 = 1
        for _ in range(2, n+1):
            now = 2 * pre_1
            pre_1 = now

        return now 

###矩形覆盖 #####################################
##可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。
##请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

http://thyrsi.com/t6/364/1535427462x-1404817880.png
##从这张图中可以非常清晰地看出，从第3步开始，矩形的叠法就只有2种
##一种是：  上一次叠了(2 * n-1)，则需横着叠1个；
##另一种是：上一次叠了(2 * n-2)，则需竖着叠2个；
##因此，f(n) = 1                if n=1
##		     = 2                if n=2
##           = f(n-1) + f(n-2)  if n>2
class Solution:
    def rectCover(self, n):
        if n <= 2:
            return n

        pre_2 = 1
        pre_1 = 2
        for _ in range(3, n+1):
            now = pre_1 + pre_2
            pre_2 = pre_1
            pre_1 = now

        return now 

    

#######################################################
##数值的整数次方 
##给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

##连乘不就可以了吗？很简单啊！！
class Solution:
    def Power(self, base, exponent):
        if exponent == 0:
            return 1
        elif exponent > 0:
            res = 1
            for _ in range(exponent):
                res *= base
            return res
        elif exponent < 0:
            res = 1
            for _ in range(-exponent):
                res *= base
            return 1 / res

#####################################################
##二进制中1的个数 
##输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

## flag <<= 1；表示向左移动一位，如：0001 -> 0010
## 假设flag = 0000 0010，则 "if (n & flag):" 表示看看flag中为1的那位（第2位），在n中是否为1
## 因为：(1100 & 0010) = (0000)，该数为0，所以第2位是0
##       (1100 & 0100) = (0100)，该数不为0，所以第3位是1

## 这里牛客网上，n默认是32位；只看32位中1的个数；
##可以不用管补码，反正统计1的个数就行了
class Solution:
    def NumberOf1(self, n):
        count = 0
        flag = 1
        
        while True:
            if (n & flag):  #flag只有一位是1，其他为0
                count += 1
            flag <<= 1
            
            if flag > (2 ** 31):  #表示flag现在已经移动到第32位(而且已经计数完32位)，则退出计数
                break
        return count
        
        
#####################################################
##数组中出现次数超过一半的数字
##数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。
##例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

##如果有符合条件的数字，则它出现的次数比其他所有数字出现的次数和还要多。 
##在遍历数组时保存两个值：一是数组中一个数字，一是次数。
##遍历下一个数字时，若它与之前保存的数字相同，则次数加1，否则次数减1；若次数为0，则保存下一个数字，并将次数置为1。
##遍历结束后，所保存的数字即为所求。然后再判断它是否符合条件即可。 
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        res = numbers[0]
        count = 1
        
        #先找出res
        for i in range(1, len(numbers)):
            if count == 0:
                res = numbers[i]
                count = 1
            elif numbers[i] == res:
                count += 1
            else:
                count -= 1
        
        #看看这个res是不是个数大于数组长度的一半
        count = 0
        for i in range(0, len(numbers)):
            if numbers[i] == res:
                count += 1
        
        if count > len(numbers) / 2:
            return res
        else:
            return 0

#######################################################
##连续子数组的最大和
##例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。

##判断以前的累计是不是>0，如果是，说明以前的累加对结果是有贡献的，就用“累加值 + 当前值”，
##如果累加值<0，说明之前的累加对结果会拖累，则舍弃累加值，从当前值开始计算。
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        if len(array) == 0:
            return None
            
        maxSum = array[0]
        accumSum = array[0]   
        for idx in range(1, len(array)):
            if accumSum > 0:  #如果累加和>0，说明之前的累加有贡献，用“累加值 + 当前值”
                accumSum += array[idx]
            else:             #如果累加和<0，说明之前的累加对结果拖累，则舍弃累加值，从当前值开始计算
                accumSum = array[idx]
                
            if accumSum > maxSum:  #记录此过程中最大的累加值，作为最终的输出
                maxSum = accumSum
                
        return maxSum
        
    
    
##########################################################	
###圆圈中最后剩下的数(孩子的游戏)
class Solution:
    def LastRemaining_Solution(self, n, m):
        if n <= 0 or m <= 0:
            return -1
            
        child_list = list(range(n))
        cur_idx = 0
        while len(child_list) > 1:
            #去到要出列的孩子的位置
            for _ in range(0, m-1):
                cur_idx += 1
                if cur_idx > len(child_list) - 1:
                    cur_idx = 0
            #孩子出列
            del child_list[cur_idx]
            if cur_idx > len(child_list) - 1:
                cur_idx = 0
                
        return child_list[0]
        
        
######################################
##从1到n整数中出现1的个数

##暴力法解决
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        all_num = 0
        for i in range(1, n+1):
            all_num += self.num_of_1(i)
        return all_num
        
    def num_of_1(self, n):
        num = 0
        while n > 0: 
            if n % 10 == 1:
                num += 1
            n = n / 10
        return num
        
        
############################################
##数组中只出现一次的两个数字 
###数组中有两个元素只出现一次，其他元素出现偶数次；找出这两个数   

##暴力法总该会吧!!!
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        res = []
        for i in range(0, len(array)):
            if array[i] not in (array[0: i] + array[i+1: ]):
                res.append(array[i])
        return res
        
        

#############################################
##数字在排序数组中出现的次数
##统计一个数字在排序数组中出现的次数

##不会就暴力呗
class Solution:
    def GetNumberOfK(self, data, k):
        l_idx = self.search_Lidx(data, k)
        r_idx = self.search_Ridx(data, k)
        if l_idx ==-1:
            return 0
        return r_idx - l_idx + 1
                        
    def search_Lidx(self, data, k):
        l_idx = 0
        for _ in range(0, len(data)):
            if data[l_idx] == k:
                return l_idx
            l_idx += 1
        return -1
       
    def search_Ridx(self, data, k):
        r_idx = len(data) - 1
        for _ in range(0, len(data)):
            if data[r_idx] == k:
                return r_idx
            r_idx -= 1
        return -1    
        

##用二分法分别找到最左边的位置和最右边的数字，
##结果为:r_idx - lidx + 1
class Solution:
    def GetNumberOfK(self, data, k):
        l_idx = self.search_Lidx(data, k)
        r_idx = self.search_Ridx(data, k)
        print(l_idx)
        print(r_idx)
        if l_idx ==-1:
            return 0
        return r_idx - l_idx + 1
            
            
    def search_Lidx(self, data, k):
        l_idx = 0
        r_idx = len(data) - 1
        
        while l_idx <= r_idx:
            m_idx = (l_idx + r_idx) // 2
            if data[m_idx] < k:
                l_idx = m_idx + 1
            elif data[m_idx] > k:
                r_idx = m_idx - 1
            elif (data[m_idx] == k) and (m_idx > 0 and data[m_idx-1] == k):
                r_idx = m_idx - 1
            elif data[m_idx] == k and (m_idx == 0  or data[m_idx-1] != k):
                return m_idx
        return -1
        
    def search_Ridx(self, data, k):
        l_idx = 0
        r_idx = len(data) - 1
        
        while l_idx <= r_idx:
            m_idx = (l_idx + r_idx) // 2
            if data[m_idx] < k:
                l_idx = m_idx + 1
            elif data[m_idx] > k:
                r_idx = m_idx - 1
            elif data[m_idx] == k and (m_idx < len(data)-1 and data[m_idx+1] == k):
                l_idx = m_idx + 1
            elif data[m_idx] == k and (m_idx == len(data)-1 or data[m_idx+1] != k):
                return m_idx
        return -1
        
        
###########################################
##数据流中的中位数             

##使用list，从尾部遍历找到最小点进行插入；
##查找时直接找中位数
##时间O(n), 空间O(n)
class Solution:
    def __init__(self):
        self.tmp_list = []
    
    def Insert(self, num):
        if self.tmp_list == []:
            self.tmp_list.append(num)
        else:
            r_idx = len(self.tmp_list) - 1
            for r_idx in range(len(self.tmp_list)-1, -1, -1):
                if num >= self.tmp_list[r_idx]:
                    self.tmp_list.insert(r_idx+1, num)
                    break
                if r_idx == 0:
                    self.tmp_list.insert(0, num)
                
    def GetMedian(self, fucking):
        if len(self.tmp_list) % 2 == 0:
            m_idx1 = len(self.tmp_list) // 2
            m_idx2 = m_idx1 - 1
            return (self.tmp_list[m_idx1] + self.tmp_list[m_idx2]) / 2.0
        else:
            m_idx = len(self.tmp_list) // 2 
            return self.tmp_list[m_idx]
            
            
##########################################            
##数组中第一个重复的数字            
##在一个长度为n的数组里， //////所有数字都在0到n-1的范围内。 
##例如长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

##至少作弊的方法写得出来吧
class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):
        for i in range(0, len(numbers)):
            if numbers[i] in numbers[0: i] + numbers[i+1: ]:
                duplication = numbers[i]
                return True
        return False
        
        
'''
非常好的方法：构造一个容量为N的辅助数组B，原数组A中每个数对应B中下标，首次命中，B中对应元素+1。
如果第二次命中时，B中对应的不为0，说明前边已经有一样数字了，那它就是重复的了。 

举例：A{1,2,3,3,4,5}，刚开始B是{0,0,0,0,0,0}，开始扫描A。 
A[0] = 1  {0,1,0,0,0,0} 
A[1] = 2  {0,1,1,0,0,0} 
A[2] = 3  {0,1,1,1,0,0} 
A[3] = 3  {0,1,1,2,0,0}，到这一步，就已经找到了重复数字。 
A[4] = 4  {0,1,1,2,1,0} 
A[5] = 5  {0,1,1,2,1,1} 
'''

class Solution:
    # 这里要特别注意~找到任意重复的一个值并赋值到duplication[0]
    # 函数返回True/False
    def duplicate(self, numbers, duplication):     
        assist_list = [0] * len(numbers)
        
        for i in range(0, len(numbers)):
            idx = numbers[i]
            assist_list[idx] += 1
            
            if assist_list[idx] > 1:
                duplication[0] = idx
                return True
        return False
        
        
###############################################
##最小的前k个数        
##输入n个整数，找出其中最小的K个数。
##例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

##练习一下快速排序和归并排序吧!!!这样做也可以了
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if k > len(tinput) or len(tinput) == 0:
            return []
        self.quick_sort(tinput, 0, len(tinput)-1)
        return tinput[0: k]
        
    def quick_sort(self, array, l_idx, r_idx):
        def parttition(array, l_idx, r_idx):
            base_value = array[l_idx]
            while l_idx < r_idx:  ##这里边界条件特别重要，是 >= ，一定要有 = 
                while (array[r_idx] >= base_value) and (l_idx < r_idx):
                    r_idx -= 1
                if l_idx < r_idx:
                    array[l_idx] = array[r_idx]  ##这里边界条件特别重要，是<=
                while (array[l_idx] <= base_value) and (l_idx < r_idx):
                     l_idx += 1
                if l_idx < r_idx:
                    array[r_idx] = array[l_idx]
            array[l_idx] = base_value
            return l_idx
            
        if l_idx < r_idx:
            part_idx = parttition(array, l_idx, r_idx)
            self.quick_sort(array, l_idx, part_idx-1)
            self.quick_sort(array, part_idx+1, r_idx)
            


class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if k > len(tinput) or len(tinput) == 0:
            return []
        array = self.merge_sort(tinput)
        return array[0: k]
        
    def merge_sort(self, array):
        def merge(l_arr, r_arr):
            l_arr_idx = 0
            r_arr_idx = 0
            merge_arr = []
            while (l_arr_idx <= len(l_arr)-1) and (r_arr_idx <= len(r_arr)-1):
                if l_arr[l_arr_idx] < r_arr[r_arr_idx]:
                    merge_arr.append(l_arr[l_arr_idx])
                    l_arr_idx += 1
                else:
                    merge_arr.append(r_arr[r_arr_idx])
                    r_arr_idx += 1
            merge_arr += l_arr[l_arr_idx: ]
            merge_arr += r_arr[r_arr_idx: ]
            return merge_arr
            
        if len(array) == 1:
            return array
            
        m_idx = len(array) // 2
        
        l_arr = self.merge_sort(array[0: m_idx])
        r_arr = self.merge_sort(array[m_idx:  ])
        return merge(l_arr, r_arr)
        
        
##设置一个stack用来保存数组的前k个元素，然后扫面数组后面的数字，如果这个数字小于max(stack)，则用这个数字替换已有的最大值。
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if k > len(tinput) or k <=0 or len(tinput) == 0:
            return []
            
        stack = tinput[0: k]
        
        for e in tinput[k: ]:
            if e < max(stack):
                stack.remove(max(stack))
                stack.append(e)
                
        return sorted(stack)
        
            
##对上面的程序优化找stack的最大值的部分，用最大堆实现：
import heapq  ##善用轮子
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if k > len(tinput) or k <=0 or len(tinput) == 0:
            return []
        
        max_heap = []
        for e in tinput[0: k]:
            heapq.heappush(max_heap, -e)  ##原始是构建最小堆，数组的每个(元素)*(-1)，变成构造最大堆
        
        for e in tinput[k: ]:
            if -e > max_heap[0]:
                heapq.heapreplace(max_heap, -e)
                
        return sorted(list(map(lambda x: -x, max_heap)))  ##将负数还原成正数
        
        
 
        
##########################################################		
##和为s的两个数字
##输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S

##夹逼准则（其实二分法也是夹逼准则）
##首先是第一个数和最后一个数相加，如果还小，则第一个数右移1位
##如果大，则右边的数左移1位
class Solution:
    def FindNumbersWithSum(self, array, tsum):
        l_idx = 0
        r_idx = len(array) - 1
        
        while l_idx < r_idx:
            if (array[l_idx] + array[r_idx]) < tsum:
                l_idx += 1
            elif (array[l_idx] + array[r_idx]) > tsum:
                r_idx -= 1
            else:
                return [array[l_idx], array[r_idx]]
        
        return []
        
        
##########################################################	
###和为S的连续正数序列 
##小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。
##但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。
##没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。
##现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

##暴力法
class Solution:
    def FindContinuousSequence(self, tsum):
        res = []
        cur_sum = 0
        for l_idx in range(1, tsum):
            if l_idx + l_idx + 1 > tsum:
                break 
            for r_idx in range(l_idx, tsum):
                cur_sum += r_idx
                if cur_sum > tsum:
                    cur_sum = 0
                    break
                elif cur_sum == tsum:
                    tmp_list = list(range(l_idx, r_idx+1))
                    res.append(tmp_list)
        return res

    


##########################################################
###构建乘积数组 
##给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],
##其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

##暴力法:
from copy import copy
class Solution:
    def multiply(self, A):
        B = [1] * len(A)
        for i in range(0, len(B)):
            A_tmp = copy(A)
            del A_tmp[i]
            for e in A_tmp:
                B[i] *= e 
        return B
        
           
'''
每次相乘其实有很多的重复项，这些重复项应该缓存起来再使用
    ---------------------
 B0 | 1  | A1 | A2 | A3 |
    ---------------------
 B1 | A0 | 1  | A2 | A3 |
    ---------------------
 B2 | A0 | A1 | 1  | A3 |
    ---------------------
 B3 | A0 | A1 | A2 | 1  |
    ---------------------
'''
class Solution:
    def multiply(self, A):
        B = [1] * len(A)
        for i in range(1, len(B)):
            B[i] = B[i-1] * A[i-1]
        
        B_tmp = [1] * len(A)
        for i in range(len(B_tmp)-2, -1, -1):
            B_tmp[i] = B_tmp[i+1] * A[i+1]
            
        for i in range(0, len(B)):
            B[i] = B[i] * B_tmp[i]
        
        return B

        
        
##########################################################	
###滑动窗口的最大值：
##给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。
##例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，
##那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}；

##暴力法
class Solution:
    def maxInWindows(self, num, size):
        if num == [] or size == 0 or size > len(num):
            return []
        res = []
        win_qu = num[0: size]  ##初始化队列
        for i in range(size, len(num)):
            res.append(self.get_winQu_max(win_qu))
            win_qu.append(num[i])  ##入队
            win_qu.pop(0)  ##出队，先忽略pop(0)的复杂的度
        res.append(self.get_winQu_max(win_qu))
        return res

    def get_winQu_max(self, win_qu):
        max_val = win_qu[0]
        for i in range(1, len(win_qu)):
            if max_val < win_qu[i]:
                max_val = win_qu[i]
        return max_val
        
    
    
######################################        
##扑克牌中的顺子(判断抽到的5张牌是不是顺子)

##1、对数组排序；2、统计0出现的个数；3、统计空缺的总数
##非0的项重复说明存在对子，返回False；如果空缺的总数<=0的个数，判断为True，否则为False
class Solution:
    def IsContinuous(self, numbers):
        if len(numbers) != 5:
            return False
        #排序
        numbers.sort()
        
        #统计0出现的个数
        count_of_0 = numbers.count(0)
        
        #统计空缺的总数
        count_of_nan = 0
        for idx in range(0, len(numbers)-1):
            if numbers[idx] > 0:
                count_of_nan += (numbers[idx+1] - numbers[idx] - 1)
                #判断是否存在对子
                if numbers[idx] == numbers[idx+1]:
                    return False
        if count_of_nan <= count_of_0:
            return True
        else:
            return False
    
    
#########################################
##把字符串转化为整数

##if c in num_list属于作弊
class Solution:
    def StrToInt(self, s):
        if s == "":
            return 0
        
        flag = 1
        if s[0] == '+':
            flag = 1
            s = s[1: ]
        elif s[0] == '-':
            flag = -1
            s = s[1: ]
        
        #主体的解法
        num_list = ['0','1','2','3','4','5','6','7','8','9']
        res = 0
        for c in s:
            if c in num_list:
                res = res * 10 + num_list.index(c)
            else:
                return 0
        res *= flag
                
        return res  
    
    
##直接用循环判断吧!            
class Solution:
    def StrToInt(self, s):
        if s == "":
            return 0
        
        flag = 1
        if s[0] == '+':
            flag = 1
            s = s[1: ]
        elif s[0] == '-':
            flag = -1
            s = s[1: ]
                    
        num_list = ['0','1','2','3','4','5','6','7','8','9']
        res = 0
        for c in s:
            for i in range(0, len(num_list) + 1):
                if i == len(num_list):
                    return 0
                if c == num_list[i]:       
                    res = res * 10 + i
                    break                
        res *= flag
                
        return res                  
    
    
    

    
    
        
        