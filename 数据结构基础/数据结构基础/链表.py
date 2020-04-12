'''
链表的链接域一定要理解为有箭头的连接，跟指针一样的指向，头脑中要想象出箭头！！！
不能理解为包含，否则复杂的问题无法形象化解释
'''
###单链表节点
class LNode():
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_

###单链表########################################################################
###O(1)：表头插入、表头删除，O(n)：表尾插入、表尾删除
class LList():
    def __init__(self):
        self._pHead = None  #用表头表示整个列表。。先初始化列表为空
        
    def is_empty(self):
        return self._pHead is None
    
    ##表头插入
    def prepend(self, elem):  
        self._pHead = LNode(elem, self._pHead)  #要理解为指针，链接到原来的表；直接表示创建节点时的链接
       
    ##表尾插入
    def append(self, elem):  
        if self._pHead is None: #空表
            self._pHead = LNode(elem, None)
            return
        _pNode = self._pHead  ###这个非常精髓，python是引用机制，不是复制机制，也就是这两个变量共享同一个内存
        while _pNode.next is not None:
            _pNode = _pNode.next
        _pNode.next = LNode(elem, None)
        
    ##表头删除
    def pop(self):  #删除首节点，并返回该节点数据
        if self._pHead is None:
            return str("空链表")
        e = self._pHead.elem
        self._pHead = self._pHead.next  #删除表头
        return e
        
    ##表尾删除
    def pop_last(self):  #后端删除
        if self._pHead is None:
            return str("空链表")
        if self._pHead.next is None:  #只有一个元素
            e = self._pHead.elem
            self._pHead = None
            return e
        _pNode = self._pHead
        while _pNode.next.next is not None:
            _pNode = _pNode.next
        e = _pNode.next.elem
        _pNode.next = None  #删除表尾
        return e
        
    ##链表遍历
    def for_each(self, proc):  #表的遍历  proc一般为print()
        _pNode = self._pHead  #遍历前先保存“引用”，然后使用引用操作
        while _pNode:
            proc(_pNode.elem)
            _pNode = _pNode.next  #删除表头
    
    ##链表打印
    def print_all(slef):
        _pNode = self._pHead
        while _pNode is not None:
            print(_pNode.elem, end=',')
            _pNode = _pNode.next
                
    
    ##链表反转
    def reverse(self):
        if self._pHead is None or self._pHead.next is None:
            return self._pHead
        
        pLast = None
        _pNode = self._pHead  #不要轻易删除原链表，遍历要在保存的另一个“引用”上操作
        while _pNode:
            pTemp = _pNode
            _pNode = _pNode.next  #删除节点
            pTemp.next = pLast
            pLast = pTemp
        return pLast
    
    ##链表排序（贼他妈复杂，就不看了吧！！！）
    def sort_bubble(self):
        if self._pHead is None or self._pHead.next is None:
            return self._pHead
        #计算pHead的长度
        _pNode = self._pHead
        len = 0
        while _pNode:
            _pNode = _pNode.next
            len += 1
        #冒泡法排序
        comp_num = len - 1  #内循环比较的次数
        for _ in range(0, len-1):
            _pNode = self._pHead
            while _pNode.next or comp_num > 0:
                if _pNode.elem > _pNode.next.elem:
                    temp = _pNode.elem   #当前节点和后一个节点交换elem
                    _pNode.elem = _pNode.next.elem
                    _pNode.next.elem = temp
                _pNode = _pNode.next  
            comp_num -= 1  #每进行一次for循环，内循环的while需要比较的次数 -1
        return self._pHead
        
    def sort_select(self):
        if self._pHead is None or self._pHead.next is None:
            return self._pHead
        #计算pHead的长度
        _pNode = self._pHead
        len = 0
        while _pNode:
            _pNode = _pNode.next
            len += 1
        #选择排序
        pTemp = self._pHead
        for _ in range(0, len-1):
            _pNode = pTemp  #第一个比较的节点
            min_PNode = pTemp
            while _pNode:
                if _pNode.elem < min_PNode.elem:
                    min_PNode = _pNode
                _pNode = _pNode.next
            pTemp = pTemp.next
        return _pHead
                    
            



        
###带尾节点的单链表（O(1)：表头插入、表头删除，表尾插入；O(n)：表尾删除）##########################################
###O(1)：表头插入、表头删除，表尾插入；O(n)：表尾删除
class rLList():
    def __init__(self):
        self._pHead = None  #用表头表示整个列表。。先初始化列表为空
        self._pRear = None
        
    def is_empty(self):
        return self._pHead is None
    
    def prepend(self, elem):  #表头插入
        if self._pHead is None:
            self._pHead = LNode(elem, None) 
            self._pRear = LNode(elem, None) 
        else:
            self._pHead = LNode(elem, self._pHead)  #直接表示创建节点时的链接
    
    def append(self, elem):  #后端插入
        if self._pHead is None: #空表
            self._pHead = LNode(elem, None)  
            self._pRear = LNode(elem, None)  
        else:
            self._pRear.next = LNode(elem, None)  #精髓
            self._pRear = self._pRear.next  ###pRear重新记录表尾
            
    def pop(self):  #删除首节点，并返回该节点数据
        if self._pHead is None:
            return str("空链表")
        e = self._pHead.elem
        self._pHead = self._pHead.next  #删除表头
        return e
       
    def pop_last(self):  #后端删除
        if self._pHead is None:
            return str("空链表")
        if self._pHead.next is None:  #只有一个元素
            e = self._pHead.elem
            self._pHead = None
            return e
        _pNode = self._pHead
        while _pNode.next.next:
            _pNode = _pNode.next  #pNode是倒数第二个元素
        e = _pNode.next.elem  #到这一步，pNode.next.next == None
        _pNode.next = None  #删除表尾
        self._pRear = _pNode  ###引用的思想太他妈精髓了
        return e




###循环单链表#######################################################################
###O(1)：表头插入、表头删除，表尾插入；O(n)：表尾删除
class cLList():
    def __init__(self):
        self._pRear = None
    
    def is_empty(self):
        return self._pRear is None
        
    
    
        
        
        

###双链表#####################################################################
###O(1)：表头插入、表头删除，表尾插入、表尾删除
class DLNode(self, elem, prev=None, next_=None):
    def __init__(self):
        self.elem = elem
        self.prev = prev
        self.next = next_
        
        
class DLList():
    def __init__(self):
        self._pHead = None
        self._pRear = None
        
    def is_empty(slef):
        return self._pHead is None
        
    def prepend(self, elem):
        _pNode = DLNode(elem, None, self._pHead)
        if self._pHead is None:
            self._pRear = _pNode
        else:
            _pNode.next.prev = _pNode
        self._pHead = _pNode
        
    def append(self, elem):
        _pNode = DLNode(elem, self._pRear, None)
        if self._pHead is None:
            self._pHead = _pNode
        else:
            _pNode.prev.next = _pNode
        self._pRear = _pNode
    
    def pop(self):
        if self._pHead is None:
            return str("空链表")
        e = self._pHead.elem
        self._pHead = self._pHead.next
        if self._pHead is not None:
            self._pHead.prev = None
        return e
        
    def pop_last(self):
        if self._pHead is None:
            return str("空链表")
        e = slef._pRear.elem
        self._pRear = self._pRear.prev
        if self._pRear is None:
            self._pHead = None
        else:
            self._pRear.next = None
        return e
    
    
    
        