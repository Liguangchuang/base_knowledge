#########################################################
##�������к�Ϊĳһֵ��·�� 
##����һ�Ŷ������ĸ��ڵ��һ����������ӡ���������н��ֵ�ĺ�Ϊ��������������·����
##·������Ϊ�����ĸ���㿪ʼ����һֱ��Ҷ����������Ľ���γ�һ��·��

##�����Ҫ�Եݹ���ڲ����˽���ܽ�����������
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
        if is_leaf:  ##������˵�ʱ�򣬰���Щ·��Ҳ����ȥ
            if sum(self.one_path[:]) == expectNumber:
                self.all_path.append(self.one_path[:])  ##[:]�����һ��Ҫ�ӵģ���ס�ˣ���
        self.one_path.pop()
        
#########################################################
##����������һ���ڵ�
##����һ�������������е�һ����㣬���ҳ��������˳�����һ����㲢�ҷ��ء�ע�⣬
##���еĽ�㲻�����������ӽ�㣬ͬʱ����ָ�򸸽���ָ�롣
#��һ��һ��Ҫע�⣬python�ĳ����У������ڵ㡱�ȼ��ڡ�����
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
##����ĳ��������ǰ���������������Ľ�������ؽ����ö�������
##���������ǰ���������������Ľ���ж������ظ������֡�
##��������ǰ���������{1,2,4,7,3,5,6,8}�������������{4,7,2,1,5,3,8,6}�����ؽ������������ء�
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

        #��ÿ���������δ������ڵ㣬���ӽڵ㣬���ӽڵ�
        pRoot = TreeNode(root_value)
        pRoot.left = self.reConstructBinaryTree(pre[1: tin_root_index+1], tin[0: tin_root_index])
        pRoot.right = self.reConstructBinaryTree(pre[tin_root_index+1: ], tin[tin_root_index+1: ])
        return pRoot

        
#########################################################
###�Ѷ�������ӡ�ɶ��� 
##���ϵ��°����ӡ��������ͬһ����������������ÿһ�����һ�С�

##һ��Ҫ����˽�python�����û��ƣ���=����list��df������...���ṹ������ʱ���Ǹ�ֵ���������ã��ǳ���Ҫ������
##��׼���ǿ���ֱ��import�ģ�����ȫ�������
##����pandas��������ʱ�����ʹ��df_ = df.copy()����������׳���������
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from copy import copy
class Solution:
    # ���ض�ά�б�[[1,2],[4,5]]
    def Print(self, pRoot):
        if pRoot is None: 
            return []

        #��ʼ��
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
##�����������ĺ���������� 
##����һ���������飬�жϸ������ǲ���ĳ�����������ĺ�������Ľ����
##����������Yes,�������No���������������������������ֶ�������ͬ��

##���ݺ�������Ĺ��ɣ������е����һ�����������ĸ��ڵ� ��
##���������������Ĺ��ɣ�������ǰ������ֿ��Է�Ϊ�����֣���һ�������������ڵ��ֵ�����ȸ��ڵ��ֵС���ڶ��������������ڵ��ֵ�����ȸ��ڵ��ֵ��
##�����õݹ�ֱ��ж�ǰ���������Ƿ��������ԭ��
class Solution:
    def VerifySquenceOfBST(self, sequence):
        if len(sequence) == 0:
            return False
        
        length = len(sequence)
        root_value = sequence[length - 1]
        
        # �ҳ����ӽڵ���ٽ�㣺�����������У��������ڵ�С�ڸ��ڵ�
        for i in range(length):
            if sequence[i] > root_value:
                break
                
        # ���ҽڵ��У����Ƿ������������Ҫ��
        # �����������У��������Ľڵ㶼���ڸ��ڵ�
        for j in range(i, length):
            if sequence[j] < root_value:
                return False
                
        # �ж��������Ƿ�Ϊ������
        left = True
        if i > 0:  # >0 �������ӽڵ㣬����ֱ���ж�ΪTrue
            left = self.VerifySquenceOfBST(sequence[0: i])
            
        # �ж��������Ƿ�Ϊ������
        right = True
        if i < length-1:  #<length-1 ����������������ֱ���ж�ΪTrue
            right = self.VerifySquenceOfBST(sequence[i: length-1])  #���������ڵ�
        
        return left and right
    
    
#########################################################
###����������� 
##����һ�ö����������������ȡ��Ӹ���㵽Ҷ������ξ����Ľ�㣨������Ҷ��㣩�γ�����һ��·����
##�·���ĳ���Ϊ�������
class Solution:
    def TreeDepth(self, pRoot):
        if pRoot is None:
            return 0
        depth_left = self.TreeDepth(pRoot.left)
        depth_right = self.TreeDepth(pRoot.right)
 		return max(depth_left, depth_right) + 1   
            
            
#########################################################
##�ж�һ�����Ƿ�Ϊƽ������� 
##����һ�ö��������жϸö������Ƿ���ƽ���������

##ƽ�����������������ڵ㣬1��������������������ƽ�������
##							2�����������������ĸ߶Ȳ�ľ���ֵ���1�����͵ĵݹ�ṹ
##���ݹ����Ŀ���м��о��ײ�ϸ�ڣ�ֻҪ���ա��ϲ������ʵ�֡��͡��˳��������Ϳ����ˣ�ϸ�ڽ������������
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

##����������������������ʱ�����ֶ�α�����Ч�ʱȽϵ�
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
###�������ľ���
'''
���������Ķ�����������任ΪԴ�������ľ���

��������:
�������ľ����壺Դ������ 
            8
           /  \
          6   10
         / \  / \
        5  7 9 11
        ���������
            8
           /  \
          10   6
         / \  / \
        11 9 7  5
'''

###�����е������ӽڵ㽻��������
class Solution:
    # ���ؾ������ĸ��ڵ�        
    def Mirror(self, root):
        if root is None:
            return 
            
        root.right, root.left = root.left, root.right  #���������ӽڵ�
        
        #�ݹ飬����ÿ��������Ҳ���������ӽڵ㣡��
        self.Mirror(root.left)
        self.Mirror(root.right)
        
        

#########################################################
###�����������ĵ�kС��� 
##����һ�ö��������������ҳ����еĵ�kС�Ľ�㡣
##���磬��5��3��7��2��4��6��8���У��������ֵ��С˳�����С����ֵΪ4��

##������ʹ�ñ���������������һ�������ڼӡ�self.������self.res = []����self.res�����ȫ���ڶ�����ʹ��
##����ȫ��������ʹ�ã�Ҫ�ӡ�self.��
##�������ں�����Ҫ�ӡ�self.���������ݹ���ã��磺self.inOrder_Traverse(pRoot.left)
# -*- coding:utf-8 -*-
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # ���ض�Ӧ�ڵ�TreeNode
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
## �������´�ӡ������
##�������´�ӡ����������ÿ���ڵ㣬ͬ��ڵ�������Ҵ�ӡ��

# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    # ���ش��ϵ���ÿ���ڵ�ֵ�б�����[1,2,3]
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
###�ԳƵĶ����� 
##��ʵ��һ�������������ж�һ�Ŷ������ǲ��ǶԳƵġ�
##ע�⣬���һ��������ͬ�˶������ľ�����ͬ���ģ�������Ϊ�ԳƵġ�
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
            return True  #���յ��ж�
        elif (p1 != None and p2 != None) and (p1.val == p2.val):
            return self.is_subTree_symmetry(p1.left, p2.right) and self.is_subTree_symmetry(p1.right, p2.left)
        else: 
            return False  #���յ��ж�
            
            
            
###########################################
##��β��ͷ��ӡ����
##����һ������������ֵ��β��ͷ��˳�򷵻�һ��ArrayList��

# -*- coding:utf-8 -*-
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # ���ش�β����ͷ�����б�ֵ���У�����[1,2,3]
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
    # ���ش�β����ͷ�����б�ֵ���У�����[1,2,3]
    def printListFromTailToHead(self, listNode):
        if listNode is None:
            return print_list
            
        print_list = []
        stack = []  #�Ƚ�ջ�����ջ�������Ȼ�͵������ˣ���
        
        while listNode:
            stack.append(listNode.val)
            listNode = listNode.next
            
        while stack:
            print_list.append(stack.pop())
        
        return print_list			
        
#########################################################
##�����е�����k����� 
##����һ����������������е�����k����㡣

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
##��õĽⷨ��
##��������ָ�룬��һ��ָ������k����Ȼ������ָ��ͬʱ�ߣ�
##��һ��ָ�����ʱ���ڶ���ָ��ǡ���ߵ�������k����
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
        

##���α������ȼ�¼һ����num���ڵ㣬����k���ڵ㣬�ȼ���ɾ��ǰnum-k�ͽڵ�
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
    
###python�Ľⷨ
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
##��ת����
##����һ��������ת��������������ı�ͷ��
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
            pTemp = pNode.next  #�ȰѺ����ڵ㻺������
            pNode.next = pLast  #��һ���ڵ��������ڵ㣻����ԭ���������Զ��Ͽ�
            pLast = pNode  #�������ڵ�
            pNode = pTemp  #�����׽ڵ�
        return pLast
            
        
    

#########################################################	
###�����л�����ڽ�� 
##��һ�����������а����������ҳ�������Ļ�����ڽ�㣬�������null��


###���Ӽ�ֱ��Ҫ̫�������
'''
ͼƬ��https://uploadfiles.nowcoder.net/images/20170422/943729_1492841744777_3BB680C9CBA20442ED66C5066E1F7175
˼·��
1����������ָ��fast��slow��fast��slow�����ٶ�ǰ����
2�����û�л�����ôfast��slow����������ʱ����None��
3������л�����fast��slow�϶����ٴ�����; 
   ������ʱ��fast�պñ�slow������һȦ���ĳ��ȣ�fast�߹��ľ���Ϊa + b + c + b����slow�߹��ľ���Ϊa + b��
   ��Ϊfast��slow�ٶȵ�����������a+b+c+b = 2*(a+b)�����a=c;
   ��ˣ����õ�����ָ��p����X�����Ժ�slowָ����ͬ���ٶ�ǰ��������������ʱ����Ϊ�������Y����
'''
class Solution:
    def EntryNodeOfLoop(self, pHead):
        if pHead ==None or pHead.next == None:
            return None
        
        pFast = pHead
        pSlow = pHead
        while pFast.next.next:
            pFast = pFast.next.next  ##while����ġ�����next���ж���Ҫ��Ϊ�����
            pSlow = pSlow.next
            if pFast == pSlow:
                newNode = pHead
                while newNode != pSlow:
                    newNode = newNode.next
                    pSlow = pSlow.next
                return newNode
        return None
                
    
    

#########################################################	
##��������ĵ�һ�������ڵ�
##�������������ҳ����ǵĵ�һ��������㡣

# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        if pHead1 is None or pHead2 is None:
            return None
        
        #�ֱ������������ĳ���
        p1, p2 = pHead1, pHead2
        length1 = length2 = 0
        while p1:
            p1 = p1.next
            length1 += 1
        while p2:
            p2 = p2.next
            length2 += 1
        
        #���ó�����(��-��)����ֱ����������ĳ���һ��
        p1, p2 = pHead1, pHead2
        if length1 > length2:
            while length1 > length2:
                p1 = p1.next
                length1 -= 1
        else:
            while length2 > length1:
                p2 = p2.next
                length2 -= 1
        
        #����������һ�µ�ʱ��ͬʱ��ֱ��������ͬ��
        while p1 != p2:
            p1 = p1.next
            p2 = p2.next
        
        return p1
                
    

#########################################################	
####�ϲ�������������� 
##���������������������������������ϳɺ������
##��Ȼ������Ҫ�ϳɺ���������㵥����������
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    # ���غϲ����б�
    def Merge(self, pHead1, pHead2):
        #�ݹ�Ľ����������ݹ�һ��Ҫ�Ӵ�ķ�����˼��
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
##��������ĸ��� 
##����һ����������ÿ���ڵ����нڵ�ֵ���Լ�����ָ�룬һ��ָ����һ���ڵ㣬
##��һ������ָ��ָ������һ���ڵ㣩�����ؽ��Ϊ���ƺ��������head

'''
Python�����C��һ���ǳ�����ĵط����������ǡ����á�����
���ֱ�ʾһ������ı���������list������ȣ���ֵ������a_list = b_list����ʾͬһ�����������������ֱ�ʾ��
����һ���������в���ʱ������������ֵ����ı䣨����a_listԭ����[1,2,3]��a_list.append(2)����ʱ��a_list��b_listͬʱ��Ϊ[1,2,3,2]��

��������pNode = pHead��Ȼ���pNode���������Ľ��pNode = None������pHead����˸�����2����������
��Ϊͷ�ڵ㴦�ġ�������һ���ģ��㲻�ϵ�ͨ������pNode���ı䡰��Щ��������ӹ�ϵ��������pHead�ġ���ʼ���򡱳�������Ȼ��õ�һ���µ�����
�ⲻ��Python�����û��ƣ��ǡ����ӹ�ϵ���ĸı䣬����ͷ�����򲻱䣬�����ӹ�ϵ���ˣ�����
'''
# class RandomListNode:
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution:
    # ���� RandomListNode
    def Clone(self, pHead):
        if pHead is None:
            return None

        pDoubleHead = pHead

        pDoubleHead = self.cloneNodes(pDoubleHead)
        pDoubleHead = self.connectRandomNodes(pDoubleHead)
        pCloneHead = self.reConnetNode(pDoubleHead)

        return pCloneHead
        
    #���ƽڵ㣬����ǰ��������
    def cloneNodes(self, pDoubleHead):
        pNode = pDoubleHead
        while pNode:
            pCloneNode = RandomListNode(pNode.label)
            pCloneNode.next = pNode.next  #�Ⱥ�˳�򲻿ɵߵ�
            pNode.next = pCloneNode
            pNode = pNode.next.next
        return pDoubleHead
        
    #��¡�ڵ��randomָ������
    def connectRandomNodes(self, pDoubleHead):
        pNode = pDoubleHead
        while pNode:
            if pNode.random:
                pNode.next.random = pNode.random.next
            pNode = pNode.next.next
        return pDoubleHead
        
    #˫������ �Ͽ�Ϊ  ԭ���� �� ��������
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
##������ջ��ʵ��һ�����У���ɶ��е�Push��Pop������ �����е�Ԫ��Ϊint����
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
##����min������ջ 
##����ջ�����ݽṹ�����ڸ�������ʵ��һ���ܹ��õ�ջ��������СԪ�ص�min����
##��ʱ�临�Ӷ�ӦΪO��1������
class Solution:
    def __init__(self):
        self.stack = []
        self.top_min_stack = []
        
    def push(self, node):
        self.stack.append(node)
        
        if (self.top_min_stack == []) or (node < self.top_min_stack[-1]):  ##����������ˣ��������˳��ͱ���
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
##ջ��ѹ�롢�������� 
##���������������У���һ�����б�ʾջ��ѹ��˳�����жϵڶ��������Ƿ����Ϊ��ջ�ĵ���˳��
##����ѹ��ջ���������־�����ȡ���������1,2,3,4,5��ĳջ��ѹ��˳������4,5,3,2,1�Ǹ�ѹջ���ж�Ӧ��һ���������У�
##��4,3,5,1,2�Ͳ������Ǹ�ѹջ���еĵ������С���ע�⣺���������еĳ�������ȵģ�

class Solution:
    def IsPopOrder(self, pushV, popV):
        stack = []
        while True:
            if pushV != []:
                stack.append(pushV.pop(0))
                
            print(pushV)
            print(popV)

            while (stack != []) and (stack[-1] == popV[0]):  ##�����ү���������˳��ͻᱨ��
                stack.pop()                                  ##һ��Ҫ��ס�����ǰ�˳���жϵģ�
                popV.pop(0)                                  ##���ж�stack != []�����ж�stack[-1] == popV[0]

            if stack == []:
                return True

            if (pushV == []) and (stack != []) and (stack[-1] != popV[0]):
                return False

                
###########################################
##˳ʱ���ӡ���� 

##��ӡ��һ�У�Ȼ��ɾ����һ�У�Ȼ��Ծ�����ת��
class Solution:
    # matrix����Ϊ��ά�б���Ҫ�����б�
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

        
##һȦһȦ��ӡ��
class Solution:
    # matrix����Ϊ��ά�б���Ҫ�����б�
    def printMatrix(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0])

        print_list = []
        start = 0

        while (cols > start * 2) and (rows > start * 2):  #��ӡ�Ľ�������
            circle_list = self.print_mat_cricle(matrix, start)
            print_list += circle_list

            start += 1

        return print_list

    def print_mat_cricle(self, matrix, start):
        circle_list = []

        end_x = len(matrix[0])-start-1
        end_y = len(matrix)-start-1

        #�������Ҵ�ӡһ��
        for c in range(start, end_x+1):
            circle_list.append(matrix[start][c])

        #�������´�ӡһ��
        if start < end_y:
            for r in range(start+1, end_y+1):
                circle_list.append(matrix[r][end_x])

        #���������ӡһ��
        if (start < end_y) and (start < end_x):
            for c in range(end_x-1, start-1, -1):
                circle_list.append(matrix[end_y][c])

        #�������ϴ�ӡһ��
        if (start < end_y-1) and (start < end_x):
            for r in range(end_y-1, start, -1):
                circle_list.append(matrix[r][start])

        return circle_list
        
        
###########################################
##�滻�ո�
##��ʵ��һ����������һ���ַ����е�ÿ���ո��滻�ɡ�%20����
##���磬���ַ���ΪWe Are Happy.�򾭹��滻֮����ַ���ΪWe%20Are%20Happy��

##��һ���µ��ַ����ϲ�����ʱ��O(n),�ռ�O(n+m)
class Solution:
    def replaceSpace(self, s):
        new_s = ''
        for c in s:
            if c == ' ':
                c = '%20'
            new_s += c
        return new_s


##��ָoffer�Ľⷨ����ԭ�����ַ����ϲ�����ʱ��O(n),�ռ�O(m)
class Solution:
    def replaceSpace(self, s):
        #ͳ�ƿո������
        l_idx = len(s) - 1
        n_space = 0
        for c in s:
            if c == ' ':
                n_space += 1

        #�����ַ���
        for _ in range(n_space):
            s += '  '  #ÿ���ո�����2��λ��
        
        #�������ַ��������
        s = list(s)    #python���ַ����ǲ��ɱ�ģ���Ҫ��ת��Ϊlist�����ܲ���
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
            
            
        
        

        
        
##�˽�һ��python�����԰ɣ�ʵ�ʲ���������
class Solution:
    # s Դ�ַ���
    def replaceSpace(self, s):
        return s.replace(' ', '%20')
        
        
        
###########################################
##�ַ����е�һ�����ظ����ַ�
##��ʵ��һ�����������ҳ��ַ����е�һ��ֻ����һ�ε��ַ���
##���磬�����ַ�����ֻ����ǰ�����ַ�"go"ʱ����һ��ֻ����һ�ε��ַ���"g"��
##���Ӹ��ַ����ж���ǰ�����ַ���"google"ʱ����һ��ֻ����һ�ε��ַ���"l"��
##�����ǰ�ַ���û�д��ڳ���һ�ε��ַ�������#�ַ���

##ʵ���ϣ��������׵���������Ϊ��ʹ�������õĹ�ϣ
class Solution:
    # ���ض�Ӧchar
    def __init__(self):
        self.s = ''
        
    def FirstAppearingOnce(self):
        for i in range(len(self.s)):
            if self.s[i] not in self.s[0: i]+ self.s[i+1: ]:
                return self.s[i]
        return '#'
        
    def Insert(self, char):
        self.s += char


##��ϣ��ʵ�֣���ϣ������ʲ��ѵ�!!
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
##��һ��ֻ����һ�ε��ַ�����λ��
##��һ���ַ������ҵ���һ��ֻ����һ�ε��ַ�,����������λ��, 
##���û���򷵻� -1
class Solution:
    def FirstNotRepeatingChar(self, s):
        for i in range(len(s)):
            if s[i] not in s[0: i] + s[i+1 :]:
                return i
        return -1 
    

##��ϣ��ʵ�֣���ʵ�ü򵥰�!!!
class Solution:
    def FirstNotRepeatingChar(self, s):
        #������ϣ��
        hashTable = [0] * 256
        for c in s:
            hashTable[ord(c)] += 1  ##����acci�빹����ϣ��
            
        for i in range(0, len(s)):
            hash_idx = ord(s[i])
            if hashTable[hash_idx] == 1:
                return i
        return -1 


#########################################        
##����ת�ַ���
##����һ���������ַ�����S���������ѭ������Kλ������������
##���磬�ַ�����S=��abcXYZdef��,Ҫ�����ѭ������3λ��Ľ��������XYZdefabc����
###�ⲻ�Ǻ��������?���ǰ�ǰnλ���ַ����ƶ���ĩβ

class Solution:
    def LeftRotateString(self, s, n):
        s_left = s[0: n]
        s_right = s[n: ]
        return s_right + s_left
        

#########################################
##��ת����˳��
##student. a am I�� --->  ��I am a student.����

##ֱ����python���õ�3���������ռ�O(n)
class Solution:
    def ReverseSentence(self, s):
        word_list = s.split(' ')
        word_list.reverse()  #ֱ�Ӹı�ԭ����list
        
        new_s = ' '.join(word_list)
        return new_s

##reverse������ջʵ��
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

##split����������ָ��ɨ��ʵ�֣�reverse�����ý���ʵ�֣�
##join������ + ����ʵ�֣��ռ�O(n)
##�Ѿ��ȽϾ�����!!
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

        
##��ָ�ı�׼��������ԭ�����ַ����ϲ������ռ�O(1)���е��Ѷ�
##�ȷ�ת��ĸ���ٷ�ת����
class Solution:
    def ReverseSentence(self, s):
        #�ȶ�������ַ���ת
        s = list(s)  #python���ַ����ǲ��ɱ�ģ���Ҫ��ת��Ϊlist�����ܲ���
        l_idx = 0
        r_idx = len(s) - 1
        while l_idx <= r_idx:
            s[l_idx], s[r_idx] = s[r_idx], s[l_idx]
            l_idx += 1
            r_idx -= 1
        
        #�ٶ�ÿ�����ʵ��ڲ��ַ���ת
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
##��ά�����еĲ��� 
##��һ����ά�����У�ÿ��һά����ĳ�����ͬ����
##ÿһ�ж����մ����ҵ�����˳������ÿһ�ж����մ��ϵ��µ�����˳������
##�����һ������������������һ����ά�����һ���������ж��������Ƿ��и�������
class Solution:
    # array ��ά�б�
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
##��ת�������С���� 
##��һ�������ʼ�����ɸ�Ԫ�ذᵽ�����ĩβ�����ǳ�֮Ϊ�������ת�� 
##����һ���Ǽ�����������һ����ת�������ת�������СԪ�ء� 
##��������{3,4,5,1,2}Ϊ{1,2,3,4,5}��һ����ת�����������СֵΪ1�� 
##NOTE������������Ԫ�ض�����0���������СΪ0���뷵��0��

##ֱ�Ӷ��ַ�����
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
##��������˳��ʹ����λ��ż��ǰ�� 
##����һ���������飬ʵ��һ�����������������������ֵ�˳��
##ʹ�����е�����λ�������ǰ�벿�֣����е�ż��λ������ĺ�벿�֣�
##����֤������������ż����ż��֮������λ�ò���

##����������ֱ��������ż��������������
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

        
###쳲�������#####################################
##f(n) = 0				if n = 0
## 	   = 1				if n = 1
##	   = f(n-1) + f(n-2)  if n > 1

##�ǵݹ�
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
        
        
##�ݹ飨���Ӷ�̫���޷����У�
##�ݹ�һ��Ҫ����㿪ʼ˼�����⣻
##����f(n-1)��f(n-2)��֪����f(n)
##��ע ���߽������� �� �����һ����Ƶ��� �Ϳ����ˣ�ʵ�ֵ�ϸ�ڽ��������
class Solution:
    def Fibonacci(self, n):
        if n <= 1:
            return n
            
        #����f(n-1)��f(n-2)��֪����f(n)
        return self.Fibonacci(n-1) + self.Fibonacci(n-2)  

###2����̨��################################################
##һֻ����һ�ο�������1��̨�ף�Ҳ��������2����
##�����������һ��n����̨���ܹ��ж������������Ⱥ����ͬ�㲻ͬ�Ľ������

##����1��̨�ף�ֻ��1�ֿ���
##����2��̨�ף������ֿ��ܣ�1��1������ֱ��2����
##...
##����n��̨�ף���һ�δ�n-1�����ģ��ǾͿ�����n-1��̨���ж����ֿ���
##		       ��һ�δ�n-2�����ģ��ǾͿ�����n-2��̨���ж����ֿ���
##���һ���Ŀ����ǣ�����n-1�������п��� + ����n-2�������п���
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

    
###n����̨��#####################################
##һֻ����һ�ο�������1��̨�ף�Ҳ��������2��������Ҳ��������n����
##�����������һ��n����̨���ܹ��ж�����������

##f(n) = f(n-1) + f(n-2) +... + f(1)      #��һ�δ�n-1��������һ�δ�n������...����һ�δ�1����
##f(n-1) = f(n-2) + f(n-3) +... + f(1)
##һʽ-��ʽΪ��f(n)=2*f(n-1)
class Solution:
    def jumpFloorII(self, n):
        if n == 1:
            return 1

        pre_1 = 1
        for _ in range(2, n+1):
            now = 2 * pre_1
            pre_1 = now

        return now 

###���θ��� #####################################
##������2*1��С���κ��Ż�������ȥ���Ǹ���ľ��Ρ�
##������n��2*1��С�������ص��ظ���һ��2*n�Ĵ���Σ��ܹ��ж����ַ�����

http://thyrsi.com/t6/364/1535427462x-1404817880.png
##������ͼ�п��Էǳ������ؿ������ӵ�3����ʼ�����εĵ�����ֻ��2��
##һ���ǣ�  ��һ�ε���(2 * n-1)��������ŵ�1����
##��һ���ǣ���һ�ε���(2 * n-2)���������ŵ�2����
##��ˣ�f(n) = 1                if n=1
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
##��ֵ�������η� 
##����һ��double���͵ĸ�����base��int���͵�����exponent����base��exponent�η���

##���˲��Ϳ������𣿺ܼ򵥰�����
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
##��������1�ĸ��� 
##����һ��������������������Ʊ�ʾ��1�ĸ��������и����ò����ʾ��

## flag <<= 1����ʾ�����ƶ�һλ���磺0001 -> 0010
## ����flag = 0000 0010���� "if (n & flag):" ��ʾ����flag��Ϊ1����λ����2λ������n���Ƿ�Ϊ1
## ��Ϊ��(1100 & 0010) = (0000)������Ϊ0�����Ե�2λ��0
##       (1100 & 0100) = (0100)��������Ϊ0�����Ե�3λ��1

## ����ţ�����ϣ�nĬ����32λ��ֻ��32λ��1�ĸ�����
##���Բ��ùܲ��룬����ͳ��1�ĸ���������
class Solution:
    def NumberOf1(self, n):
        count = 0
        flag = 1
        
        while True:
            if (n & flag):  #flagֻ��һλ��1������Ϊ0
                count += 1
            flag <<= 1
            
            if flag > (2 ** 31):  #��ʾflag�����Ѿ��ƶ�����32λ(�����Ѿ�������32λ)�����˳�����
                break
        return count
        
        
#####################################################
##�����г��ִ�������һ�������
##��������һ�����ֳ��ֵĴ����������鳤�ȵ�һ�룬���ҳ�������֡�
##��������һ������Ϊ9������{1,2,3,2,2,2,5,4,2}����������2�������г�����5�Σ��������鳤�ȵ�һ�룬������2����������������0��

##����з������������֣��������ֵĴ����������������ֳ��ֵĴ����ͻ�Ҫ�ࡣ 
##�ڱ�������ʱ��������ֵ��һ��������һ�����֣�һ�Ǵ�����
##������һ������ʱ��������֮ǰ�����������ͬ���������1�����������1��������Ϊ0���򱣴���һ�����֣�����������Ϊ1��
##��������������������ּ�Ϊ����Ȼ�����ж����Ƿ�����������ɡ� 
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        res = numbers[0]
        count = 1
        
        #���ҳ�res
        for i in range(1, len(numbers)):
            if count == 0:
                res = numbers[i]
                count = 1
            elif numbers[i] == res:
                count += 1
            else:
                count -= 1
        
        #�������res�ǲ��Ǹ����������鳤�ȵ�һ��
        count = 0
        for i in range(0, len(numbers)):
            if numbers[i] == res:
                count += 1
        
        if count > len(numbers) / 2:
            return res
        else:
            return 0

#######################################################
##���������������
##����:{6,-3,-2,7,-15,1,2,2},����������������Ϊ8(�ӵ�0����ʼ,����3��Ϊֹ)��

##�ж���ǰ���ۼ��ǲ���>0������ǣ�˵����ǰ���ۼӶԽ�����й��׵ģ����á��ۼ�ֵ + ��ǰֵ����
##����ۼ�ֵ<0��˵��֮ǰ���ۼӶԽ�������ۣ��������ۼ�ֵ���ӵ�ǰֵ��ʼ���㡣
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        if len(array) == 0:
            return None
            
        maxSum = array[0]
        accumSum = array[0]   
        for idx in range(1, len(array)):
            if accumSum > 0:  #����ۼӺ�>0��˵��֮ǰ���ۼ��й��ף��á��ۼ�ֵ + ��ǰֵ��
                accumSum += array[idx]
            else:             #����ۼӺ�<0��˵��֮ǰ���ۼӶԽ�����ۣ��������ۼ�ֵ���ӵ�ǰֵ��ʼ����
                accumSum = array[idx]
                
            if accumSum > maxSum:  #��¼�˹����������ۼ�ֵ����Ϊ���յ����
                maxSum = accumSum
                
        return maxSum
        
    
    
##########################################################	
###ԲȦ�����ʣ�µ���(���ӵ���Ϸ)
class Solution:
    def LastRemaining_Solution(self, n, m):
        if n <= 0 or m <= 0:
            return -1
            
        child_list = list(range(n))
        cur_idx = 0
        while len(child_list) > 1:
            #ȥ��Ҫ���еĺ��ӵ�λ��
            for _ in range(0, m-1):
                cur_idx += 1
                if cur_idx > len(child_list) - 1:
                    cur_idx = 0
            #���ӳ���
            del child_list[cur_idx]
            if cur_idx > len(child_list) - 1:
                cur_idx = 0
                
        return child_list[0]
        
        
######################################
##��1��n�����г���1�ĸ���

##���������
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
##������ֻ����һ�ε��������� 
###������������Ԫ��ֻ����һ�Σ�����Ԫ�س���ż���Σ��ҳ���������   

##�������ܸû��!!!
class Solution:
    # ����[a,b] ����ab�ǳ���һ�ε���������
    def FindNumsAppearOnce(self, array):
        res = []
        for i in range(0, len(array)):
            if array[i] not in (array[0: i] + array[i+1: ]):
                res.append(array[i])
        return res
        
        

#############################################
##���������������г��ֵĴ���
##ͳ��һ�����������������г��ֵĴ���

##����ͱ�����
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
        

##�ö��ַ��ֱ��ҵ�����ߵ�λ�ú����ұߵ����֣�
##���Ϊ:r_idx - lidx + 1
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
##�������е���λ��             

##ʹ��list����β�������ҵ���С����в��룻
##����ʱֱ������λ��
##ʱ��O(n), �ռ�O(n)
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
##�����е�һ���ظ�������            
##��һ������Ϊn������� //////�������ֶ���0��n-1�ķ�Χ�ڡ� 
##���糤��Ϊ7������{2,3,1,0,2,5,3}����ô��Ӧ������ǵ�һ���ظ�������2��

##�������׵ķ���д�ó�����
class Solution:
    # ����Ҫ�ر�ע��~�ҵ������ظ���һ��ֵ����ֵ��duplication[0]
    # ��������True/False
    def duplicate(self, numbers, duplication):
        for i in range(0, len(numbers)):
            if numbers[i] in numbers[0: i] + numbers[i+1: ]:
                duplication = numbers[i]
                return True
        return False
        
        
'''
�ǳ��õķ���������һ������ΪN�ĸ�������B��ԭ����A��ÿ������ӦB���±꣬�״����У�B�ж�ӦԪ��+1��
����ڶ�������ʱ��B�ж�Ӧ�Ĳ�Ϊ0��˵��ǰ���Ѿ���һ�������ˣ����������ظ����ˡ� 

������A{1,2,3,3,4,5}���տ�ʼB��{0,0,0,0,0,0}����ʼɨ��A�� 
A[0] = 1  {0,1,0,0,0,0} 
A[1] = 2  {0,1,1,0,0,0} 
A[2] = 3  {0,1,1,1,0,0} 
A[3] = 3  {0,1,1,2,0,0}������һ�������Ѿ��ҵ����ظ����֡� 
A[4] = 4  {0,1,1,2,1,0} 
A[5] = 5  {0,1,1,2,1,1} 
'''

class Solution:
    # ����Ҫ�ر�ע��~�ҵ������ظ���һ��ֵ����ֵ��duplication[0]
    # ��������True/False
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
##��С��ǰk����        
##����n���������ҳ�������С��K������
##��������4,5,1,6,2,7,3,8��8�����֣�����С��4��������1,2,3,4,��

##��ϰһ�¿�������͹鲢�����!!!������Ҳ������
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if k > len(tinput) or len(tinput) == 0:
            return []
        self.quick_sort(tinput, 0, len(tinput)-1)
        return tinput[0: k]
        
    def quick_sort(self, array, l_idx, r_idx):
        def parttition(array, l_idx, r_idx):
            base_value = array[l_idx]
            while l_idx < r_idx:  ##����߽������ر���Ҫ���� >= ��һ��Ҫ�� = 
                while (array[r_idx] >= base_value) and (l_idx < r_idx):
                    r_idx -= 1
                if l_idx < r_idx:
                    array[l_idx] = array[r_idx]  ##����߽������ر���Ҫ����<=
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
        
        
##����һ��stack�������������ǰk��Ԫ�أ�Ȼ��ɨ�������������֣�����������С��max(stack)��������������滻���е����ֵ��
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
        
            
##������ĳ����Ż���stack�����ֵ�Ĳ��֣�������ʵ�֣�
import heapq  ##��������
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if k > len(tinput) or k <=0 or len(tinput) == 0:
            return []
        
        max_heap = []
        for e in tinput[0: k]:
            heapq.heappush(max_heap, -e)  ##ԭʼ�ǹ�����С�ѣ������ÿ��(Ԫ��)*(-1)����ɹ�������
        
        for e in tinput[k: ]:
            if -e > max_heap[0]:
                heapq.heapreplace(max_heap, -e)
                
        return sorted(list(map(lambda x: -x, max_heap)))  ##��������ԭ������
        
        
 
        
##########################################################		
##��Ϊs����������
##����һ����������������һ������S���������в�����������ʹ�����ǵĺ�������S

##�б�׼����ʵ���ַ�Ҳ�Ǽб�׼��
##�����ǵ�һ���������һ������ӣ������С�����һ��������1λ
##��������ұߵ�������1λ
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
###��ΪS�������������� 
##С����ϲ����ѧ,��һ����������ѧ��ҵʱ,Ҫ������9~16�ĺ�,�����Ͼ�д������ȷ����100��
##���������������ڴ�,�����뾿���ж������������������еĺ�Ϊ100(���ٰ���������)��
##û���,���͵õ���һ������������Ϊ100������:18,19,20,21,22��
##���ڰ����⽻����,���ܲ���Ҳ�ܿ���ҳ����к�ΪS��������������? Good Luck!

##������
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
###�����˻����� 
##����һ������A[0,1,...,n-1],�빹��һ������B[0,1,...,n-1],
##����B�е�Ԫ��B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]������ʹ�ó�����

##������:
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
ÿ�������ʵ�кܶ���ظ����Щ�ظ���Ӧ�û���������ʹ��
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
###�������ڵ����ֵ��
##����һ������ͻ������ڵĴ�С���ҳ����л�����������ֵ�����ֵ��
##���磬�����������{2,3,4,2,6,2,5,1}���������ڵĴ�С3��
##��ôһ������6���������ڣ����ǵ����ֵ�ֱ�Ϊ{4,4,6,6,6,5}��

##������
class Solution:
    def maxInWindows(self, num, size):
        if num == [] or size == 0 or size > len(num):
            return []
        res = []
        win_qu = num[0: size]  ##��ʼ������
        for i in range(size, len(num)):
            res.append(self.get_winQu_max(win_qu))
            win_qu.append(num[i])  ##���
            win_qu.pop(0)  ##���ӣ��Ⱥ���pop(0)�ĸ��ӵĶ�
        res.append(self.get_winQu_max(win_qu))
        return res

    def get_winQu_max(self, win_qu):
        max_val = win_qu[0]
        for i in range(1, len(win_qu)):
            if max_val < win_qu[i]:
                max_val = win_qu[i]
        return max_val
        
    
    
######################################        
##�˿����е�˳��(�жϳ鵽��5�����ǲ���˳��)

##1������������2��ͳ��0���ֵĸ�����3��ͳ�ƿ�ȱ������
##��0�����ظ�˵�����ڶ��ӣ�����False�������ȱ������<=0�ĸ������ж�ΪTrue������ΪFalse
class Solution:
    def IsContinuous(self, numbers):
        if len(numbers) != 5:
            return False
        #����
        numbers.sort()
        
        #ͳ��0���ֵĸ���
        count_of_0 = numbers.count(0)
        
        #ͳ�ƿ�ȱ������
        count_of_nan = 0
        for idx in range(0, len(numbers)-1):
            if numbers[idx] > 0:
                count_of_nan += (numbers[idx+1] - numbers[idx] - 1)
                #�ж��Ƿ���ڶ���
                if numbers[idx] == numbers[idx+1]:
                    return False
        if count_of_nan <= count_of_0:
            return True
        else:
            return False
    
    
#########################################
##���ַ���ת��Ϊ����

##if c in num_list��������
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
        
        #����Ľⷨ
        num_list = ['0','1','2','3','4','5','6','7','8','9']
        res = 0
        for c in s:
            if c in num_list:
                res = res * 10 + num_list.index(c)
            else:
                return 0
        res *= flag
                
        return res  
    
    
##ֱ����ѭ���жϰ�!            
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
    
    
    

    
    
        
        