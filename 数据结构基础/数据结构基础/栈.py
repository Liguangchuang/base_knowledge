###顺序表实现栈类###################################################
###列表尾部为“栈顶”，列表首部为“栈底”
class SStack():
    def __init__(self):
        self._elems = []
    
    def is_empty(self):
        return self._elems == []
    
    def getTop(self):  #访问栈顶元素
        if self._elems == []:
            return str("空栈")
        return self._elems[-1]
    
    def push(self, e):  #压栈：在栈顶增加一个元素
        self._elems.append(e)
        
    def pop(self):  #弹栈：删除栈顶元素，并返回
        if self._elems == []:
            return str("空栈")
        return self._elems.pop()
    
    
###单链接表实现栈类###################################################
###链接表节点类
class LNode():
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_

        
###链表的首部为"栈顶"，尾部为“栈底”
class LStack():
    def __init__(self):
        self.pTop = None
        
    def is_empty(self):
        return self.pTop is None
    
    def getTop(self):  #访问栈顶元素
        if self.pTop is None:
            return str("空栈")
        return self.pTop.elem
    
    def push(self, elem):  #压栈：在栈顶增加一个元素
        self.pTop = LNode(elem, self.pTop)
        
    def pop(self):  #弹栈：删除栈顶元素，并返回
        if self.pTop is None:
            return str("空栈")
        e = self.pTop.elem
        self.pTop = self.pTop.next
        return e

###递归与栈（很重要！！！）#############################
###阶乘的实现
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
factorial(4)


def factorial_norec(n):
    stack = []
    res = 1
    while n > 0:
        stack.append(n)
        n -= 1
    while stack != []:
        res *= stack.pop()
    return res
factorial_norec(4)


###深度模拟计算机内部的工作原理，可惜没实现
def factorial_norec_2(n):
    stack_n = []
    stack_fac = []
    stack_res = []
    
    while n >= 0:
        stack_n.append(n)
        stack_fac.append(None)
        stack_res.append(None)
        if n == 0:
            break
        n -= 1
        
    if n == 0:
        stack_res[-1] = 1
        
    while stack_n != []:
        _ = stack_n.pop()
        _ = stack_fac.pop()
        res = stack_res.pop()
        
        stack_fac[-1] = res
        stack_res[-1] = stack_n[-1] * stack_fac[-1]

    return res
factorial_norec_2(4)


