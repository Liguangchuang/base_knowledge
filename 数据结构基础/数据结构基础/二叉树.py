###���Ľڵ���################################################################
class BinTNode():
    def __init__(self, x, left=None, right=None):
        self.data = x
        self.left = left
        self.right = right

        
###ʹ�õ��ĳ���ջ�Ͷ������ݽṹ	
class SStack():
    def __init__(self):
    def is_empty(self):
    def getTop(self):  #����ջ��Ԫ��
    def push(self, elem):  #ѹջ����ջ������һ��Ԫ��
    def pop(self):  #��ջ��ɾ��ջ��Ԫ�أ�������

class SQuene():
    def __init__(self):
    def is_empty(self):
    def getHead(self):  #�鿴����Ԫ��
    def enquene(self, elem):  #����
    def dequene(self):  #����

'''
����������̫�Ѹ��ˣ�����ֱ�����ⲿд�����Ķ��壡����
'''

###�����������������ȸ�������ķ�ʽ��������
###�������������ʱ��Ҫ�����������pRoot=None�� ����������
def creat_BinTree(pRoot):  
    data = input()
    if data == '#':  #�ݹ�Ľ���
        pRoot = None
    else:
        pRoot = BinTNode(data)  ##�ݹ�һ��Ҫ�Ӵ�ķ���˼������������壬�����ϸ��
        pRoot.left = creat_BinTree(pRoot.left)
        pRoot.right = creat_BinTree(pRoot.right)
    return pRoot

    

##�������еĶ�������T��������creat_BinTree������������������
###�������Ľڵ����
def count_BinTNode(pRoot):
    if pRoot is None:
        return 0
    else:
        retuen 1 + count_BinTNode(pRoot.left) + count_BinTNode(pRoot.right)  ##�ݹ�һ��Ҫ��������������

        
###�������Ľڵ���ͣ�����ڵ㱣�������ֵ��
def sum_BinTNode(pRoot):
    if pRoot is None:
        return 0
    else:
        return pRoot.data + sum_BinTNode(pRoot.left) + sum_BinTNode(pRoot.right)

        
        
        
###�ȸ����������ǰ�������
def preOrder_Traverse(pRoot):  
    if pRoot is None:                #T���ⲿ�Ѿ����ɵĶ�����
        return                   #�ݹ��˼·��һ��Ҫ�Ӵ�С˼�����������������Ľṹ��ÿһ��С����Ҳ���������Ľṹ��
    print(pRoot.data)  #������������print()     
    preOrder_Traverse(pRoot.left) 
    preOrder_Traverse(pRoot.right)
    
########################
##���������Ĳ��ݹ���ã�
-----------  pRoot

  ---------  pRoot.left
    -------  pRoot.left.left   -> None
    -------  pRoot.left.right  -> None
    
  ---------  pRoot.right
    -------  pRoot.right.left  -> None
    -------  pRoot.right.right -> None
########################
    
#ʹ�á�listջ���ķǵݹ�ʵ��
def preOrder_Traverse_nonrec(pRoot):  
    stack = []
    bT = pRoot
    while bT or stack:
        while bT:
            print(bT.data)
            stack.append(bT)  #ջ��洢��ǰ�ڵ�
            bT = bT.left
        bT = stack.pop()  #bTΪ�գ��������ջ�е����ڵ�
        bT = bT.right  #ѡ��ǰ�����ҽڵ�

#ʹ�á�����ջ���ķǵݹ�ʵ��
def preOrder_Traverse_nonrec_2(pRoot):  
    stack = SStack()
    bT = pRoot
    while bT or not stack.is_empty():
        while bT:
            print(bT.data)
            stack.push(bT)
            bT = bT.left
        bT = stack.pop()  #bTΪ�գ��������ջ�е����ڵ�
        bT = bT.right  #ѡ��ǰ�����ҽڵ�

        
###�и�����������������
def inOrder_Traverse(pRoot):  
    if pRoot is None:
        return
    inOrder_Traverse(pRoot.left)
    print(pRoot.data)  
    inOrder_Traverse(pRoot.right)
    
#ʹ�á�listջ���ķǵݹ�ʵ��
def inOrder_Traverse_nonrec(pRoot):  
    stack = []
    bT = pRoot
    while bT or stack:
        while bT:
            stack.append(bT)  #ջ��洢��ǰ�ڵ�
            bT = pRoot.left
        bT = stack.pop()
        print(bT.data)  #�롰�ȸ���Ψһ�����������е�λ��
        bT = bT.right
        
#ʹ�á�����ջ���ķǵݹ�ʵ��
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

        
###�������������������
def postOrder_Traverse(pRoot):  
    if pRoot is None:
        return
    postOrder_Traverse(pRoot.left)
    postOrder_Traverse(pRoot.right)
    print(pRoot.data)  #��������
    
def postOrder_Traverse_nonrec(pRoot):  #����д��̫���ˣ�����
def postOrder_Traverse_nonrec_2(pRoot):

    
###��α��������У�
#ʹ�á�list���С�ʵ�֣��ص㣩
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

#ʹ�á�������С�ʵ��
def levelOrder_Traverse_2(pRoot):
    qu = SQuene()
    qu.enquene(pRoot)
    
    while not qu.is_empty():
        bT = qu.dequene()
        if bT is None:  #�����ǰ��������Ϊ�գ����������
            continue
        qu.enquene(bT.left)
        qu.enquene(bT.right)
        print(bT.data)
        

###��������
class HTNode(BinTNode):   #�̳ж������ڵ���
    def __lt__(self, otherNode):
        return self.data < otherNode.data
        
class HuffmanPrioQ(PrioQueue):  #�̳����ȶ�����
    def number(self):
        return len(self._elems)

##�������������㷨
def HuffmanTree(weights):
    H_prio_qu = HuffmanPrioQ()
    for w in weights:  
        H_prio_qu.enqueue(HTNode(w))  #ȫ���������ȶ���
    while H_prio_qu.number() > 1:
        t1 = H_prio_qu.dequeue()
        t2 = H_prio_qu.dequeue()
        x = t1.data + t2.data
        H_prio_qu.enquene(HTNode(x, t1, t2))
    return H_prio_qu.dequene()
        

###����������
##����
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
    
##����
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
    
##ɾ����