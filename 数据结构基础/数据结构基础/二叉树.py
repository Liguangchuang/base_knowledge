###树的节点类################################################################
class BinTNode():
    def __init__(self, x, left=None, right=None):
        self.data = x
        self.left = left
        self.right = right

        
###使用到的抽象栈和队列数据结构	
class SStack():
    def __init__(self):
    def is_empty(self):
    def getTop(self):  #访问栈顶元素
    def push(self, elem):  #压栈：在栈顶增加一个元素
    def pop(self):  #弹栈：删除栈顶元素，并返回

class SQuene():
    def __init__(self):
    def is_empty(self):
    def getHead(self):  #查看队首元素
    def enquene(self, elem):  #进队
    def dequene(self):  #出队

'''
二叉树的类太难搞了，现在直接在外部写函数的定义！！！
'''

###创建二叉树（按照先根序遍历的方式建立树）
###调用这个函数的时候，要传这个参数“pRoot=None” 进来！！！
def creat_BinTree(pRoot):  
    data = input()
    if data == '#':  #递归的结束
        pRoot = None
    else:
        pRoot = BinTNode(data)  ##递归一定要从大的方向思考：先设计整体，再设计细节
        pRoot.left = creat_BinTree(pRoot.left)
        pRoot.right = creat_BinTree(pRoot.right)
    return pRoot

    

##下面所有的二叉树“T”都是由creat_BinTree函数创建而来！！！
###二叉树的节点个数
def count_BinTNode(pRoot):
    if pRoot is None:
        return 0
    else:
        retuen 1 + count_BinTNode(pRoot.left) + count_BinTNode(pRoot.right)  ##递归一定要从整体出发来理解

        
###二叉树的节点求和（假设节点保存的是数值）
def sum_BinTNode(pRoot):
    if pRoot is None:
        return 0
    else:
        return pRoot.data + sum_BinTNode(pRoot.left) + sum_BinTNode(pRoot.right)

        
        
        
###先根序序遍历（前序遍历）
def preOrder_Traverse(pRoot):  
    if pRoot is None:                #T是外部已经生成的二叉树
        return                   #递归的思路：一定要从大到小思考，整体满足这样的结构，每一个小部分也满足这样的结构；
    print(pRoot.data)  #操作函数，如print()     
    preOrder_Traverse(pRoot.left) 
    preOrder_Traverse(pRoot.right)
    
########################
##遍历函数的层层递归调用：
-----------  pRoot

  ---------  pRoot.left
    -------  pRoot.left.left   -> None
    -------  pRoot.left.right  -> None
    
  ---------  pRoot.right
    -------  pRoot.right.left  -> None
    -------  pRoot.right.right -> None
########################
    
#使用“list栈”的非递归实现
def preOrder_Traverse_nonrec(pRoot):  
    stack = []
    bT = pRoot
    while bT or stack:
        while bT:
            print(bT.data)
            stack.append(bT)  #栈里存储当前节点
            bT = bT.left
        bT = stack.pop()  #bT为空，则继续从栈中弹出节点
        bT = bT.right  #选择当前树的右节点

#使用“抽象栈”的非递归实现
def preOrder_Traverse_nonrec_2(pRoot):  
    stack = SStack()
    bT = pRoot
    while bT or not stack.is_empty():
        while bT:
            print(bT.data)
            stack.push(bT)
            bT = bT.left
        bT = stack.pop()  #bT为空，则继续从栈中弹出节点
        bT = bT.right  #选择当前树的右节点

        
###中根序遍历（中序遍历）
def inOrder_Traverse(pRoot):  
    if pRoot is None:
        return
    inOrder_Traverse(pRoot.left)
    print(pRoot.data)  
    inOrder_Traverse(pRoot.right)
    
#使用“list栈”的非递归实现
def inOrder_Traverse_nonrec(pRoot):  
    stack = []
    bT = pRoot
    while bT or stack:
        while bT:
            stack.append(bT)  #栈里存储当前节点
            bT = pRoot.left
        bT = stack.pop()
        print(bT.data)  #与“先根序”唯一的区别在这行的位置
        bT = bT.right
        
#使用“抽象栈”的非递归实现
def inOrder_Traverse_nonrec_2(pRoot): 
    stack = SStack
    bT = pRoot
    while bT or not stack.is_empty:
        while bT:
            stack.push(bT)
            bT = bT.left
        bT = stack.pop()
        print(bT.data)
        bT = bT.right

        
###后根序遍历（后序遍历）
def postOrder_Traverse(pRoot):  
    if pRoot is None:
        return
    postOrder_Traverse(pRoot.left)
    postOrder_Traverse(pRoot.right)
    print(pRoot.data)  #操作函数
    
def postOrder_Traverse_nonrec(pRoot):  #不会写，太难了！！！
def postOrder_Traverse_nonrec_2(pRoot):

    
###层次遍历（队列）
#使用“list队列”实现（重点）
def levelOrder_Traverse(pRoot):
    qu = []
    qu.append(pRoot)
    
    while qu:
        bT = qu.pop(0)
        if bT is None:  
            continue
        qu.append(bT.left)
        qu.append(bT.right)
        print(bT.data)

#使用“抽象队列”实现
def levelOrder_Traverse_2(pRoot):
    qu = SQuene()
    qu.enquene(pRoot)
    
    while not qu.is_empty():
        bT = qu.dequene()
        if bT is None:  #如果当前弹出的树为空，则继续弹出
            continue
        qu.enquene(bT.left)
        qu.enquene(bT.right)
        print(bT.data)
        

###哈夫曼树
class HTNode(BinTNode):   #继承二叉树节点类
    def __lt__(self, otherNode):
        return self.data < otherNode.data
        
class HuffmanPrioQ(PrioQueue):  #继承优先队列类
    def number(self):
        return len(self._elems)

##哈夫曼树构造算法
def HuffmanTree(weights):
    H_prio_qu = HuffmanPrioQ()
    for w in weights:  
        H_prio_qu.enqueue(HTNode(w))  #全部进入优先队列
    while H_prio_qu.number() > 1:
        t1 = H_prio_qu.dequeue()
        t2 = H_prio_qu.dequeue()
        x = t1.data + t2.data
        H_prio_qu.enquene(HTNode(x, t1, t2))
    return H_prio_qu.dequene()
        

###搜索二叉树
##插入
def insert(pRoot, key):
    bT = pRoot
    if bT is None:
        pRoot = BinTNode(key)
        return 
    while True:
        cuN_key = bT.data
        if key < cuN_key:
            if bT.left is None:
                bT.left = BinTNode(key)
                return 
            bT = bT.left
        elif key > cuN_key:
            if bT.right is None:
                bT.right = BinTNode(key)
                return 
            bT = bT.right
        else:
            return 
    
##搜索
def search(pRoot, key):  
    bT = pRoot
    while bT:
        cuN_key = bT.data  
        if key < cuN_key:
            bT = bT.left
        elif key > cuN_key:
            bT = bT.right
        else: 
            return True 
    return False
    
##删除：