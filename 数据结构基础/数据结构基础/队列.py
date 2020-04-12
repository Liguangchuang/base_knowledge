###顺序表实现队列################################################
###列表头部是“队头”，列表尾部是“队尾”
class SQuene():
    def __init__(self):
        self._elems = []
    
    def is_empty(self):
        return self._elems == []
    
    def getHead(self):  #查看队首元素
        if self._elems == []:
            return str('空队列')
        return self._elems[0]
    
    def enquene(self, e):  ##进队，时间复杂度O(1)
        self._elems.append(e)
        
    def dequene(self):  ##出队，时间复杂度O(n)
        if self._elems == []:
            return str('空队列')
        return self._elems.pop(0)

        
        
        
###带尾节点的链表实现队列################################################
###O(1)操作，不错的
###链表头部为“队头”，列表尾部为“队尾”；从队尾进，从队头出
class LNode():
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_
    
class rLQuene():
    def __init__(self):
        self.pHead = None
        self.pRear = None
        
    def is_empty(self):
        return self.pHead is not None
    
    def getHead(self):  #查看队首元素
        if self.pHead is None:
            return str('空队列')
        pTemp = self.pHead
        pTemp = pTemp.next
        pTemp = None
        return pTemp.elem
    
    def enquene(self, elem):  #进队
        if self.pHead is None:
            self.pHead = LNode(elem, None)  #此时pHead=None
            self.pRear = pHead
        else:
            self.pRear.next = LNode(elem, None)
            self.pRear = self.pRear.next

        
    def dequene(self):  #出队（删除队头元素）
        if self.pHead is None:
            return str('空队列')
        e = self.pHead.elem
        self.pHead = self.pHead.next
        return e
        
        
        
        
        
###两个栈实现队列#############################################################		
###直接list实现版
class Stack_to_Quene():
    def __init__(self):
        self.stack_1 = []  #入队栈
        self.stack_2 = []  #出队栈
        
    def is_empty(self):  
        if self.stack_1 == [] and self.stack_2 == []:
            return True
        else:
            return False
            
    def getHead(self):  #查看队首元素
        if self.stack_1 == [] and self.stack_2 == []:
            return str('空队列')
        if self.stack_2 != []:
            return self.stack_2[-1]
        else:
            while len(self.stack_1) > 0:
                self.stack_2.append(self.stack_1.pop())
            return self.stack_2[-1]
            
    def enquene(self, node):  #进队
        self.stack_1.append(node)  # 列表的方法，把一个元素添加到末尾

    def dequene(self):  #出队
        if len(self.stack_2) > 0:
            return self.stack_2.pop()
        while len(self.stack_1) > 0:
            self.stack_2.append(self.stack_1.pop())
        if len(self.stack_2) == 0:
            return str('空队列')
        return self.stack_2.pop()  # 列表的方法，删除列表最后一个元素，并将该元素返回



###stack抽象版
class Stack_to_Quene_2():
    def __init__(self):
        self.stack_1 = SStack()
        self.stack_2 = SStack()
        
    def is_empty(self):  
        if self.stack_1.is_empty() and self.stack_2.is_empty():
            return True
        else:
            return False
            
    def getHead(self):  #查看队首元素
        if self.stack_1.is_empty() and self.stack_2.is_empty():
            return str('空队列')
        if not self.stack_2.is_empty():
            return self.stack_2.getHead()
        else:
            while not self.stack_1.is_empty():
                self.stack_2.push(self.stack_1.pop())
            return self.stack_2.getHead()

    def enquene(self, node):
        self.stack_1.push(node)  # 列表的方法，把一个元素添加到末尾

    def dequene(self):
        if not self.stack_2.is_empty():
            return self.stack_2.pop()
        while not self.stack_1.is_empty():
            self.stack_2.push(self.stack_1.pop())
        if self.stack_2.is_empty():
            return str('空队列')
        return self.stack_2.pop()  # 列表的方法，删除列表最后一个元素，并将该元素返回


        
###优先队列################################################################
'''
优先队列的性质：任何时候访问或者弹出的，总是队列里面优先级最高的元素
'''
###线性表优先队列：
class PrioQue():
    def __init__(self, e_list = []):
        self._elems = list(e_list)
        self._elems.sort(reverse = True)   #较小的作为较优先
        
    def enquene(self, e):  #我曹，二分查找得到插入位置不就可以了吗？
        inser_index = len(self._elems) - 1
        while inser_index >= 0:
            if self._elems[inser_index] < e:
                inser_index -= 1
            else:
                break
        self._elems.insert(inser_index + 1, e)
    
    def is_empty(self):
        return self._elems == []
    
    def getHead(self):  #查看队首元素
        if self._elems == []:
            return str('空队列')
        return self._elems[-1]
        
    def dequene(self):  ##出队，时间复杂度O(1)
        if self._elems == []:
            return str('空队列')
        return self._elems.pop()
        
###堆实现优先队列：
