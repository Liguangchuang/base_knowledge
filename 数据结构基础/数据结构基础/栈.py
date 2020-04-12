###˳���ʵ��ջ��###################################################
###�б�β��Ϊ��ջ�������б��ײ�Ϊ��ջ�ס�
class SStack():
    def __init__(self):
        self._elems = []
    
    def is_empty(self):
        return self._elems == []
    
    def getTop(self):  #����ջ��Ԫ��
        if self._elems == []:
            return str("��ջ")
        return self._elems[-1]
    
    def push(self, e):  #ѹջ����ջ������һ��Ԫ��
        self._elems.append(e)
        
    def pop(self):  #��ջ��ɾ��ջ��Ԫ�أ�������
        if self._elems == []:
            return str("��ջ")
        return self._elems.pop()
    
    
###�����ӱ�ʵ��ջ��###################################################
###���ӱ�ڵ���
class LNode():
    def __init__(self, elem, next_=None):
        self.elem = elem
        self.next = next_

        
###������ײ�Ϊ"ջ��"��β��Ϊ��ջ�ס�
class LStack():
    def __init__(self):
        self.pTop = None
        
    def is_empty(self):
        return self.pTop is None
    
    def getTop(self):  #����ջ��Ԫ��
        if self.pTop is None:
            return str("��ջ")
        return self.pTop.elem
    
    def push(self, elem):  #ѹջ����ջ������һ��Ԫ��
        self.pTop = LNode(elem, self.pTop)
        
    def pop(self):  #��ջ��ɾ��ջ��Ԫ�أ�������
        if self.pTop is None:
            return str("��ջ")
        e = self.pTop.elem
        self.pTop = self.pTop.next
        return e

###�ݹ���ջ������Ҫ��������#############################
###�׳˵�ʵ��
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


###���ģ�������ڲ��Ĺ���ԭ����ϧûʵ��
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


