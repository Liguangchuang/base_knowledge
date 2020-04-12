###˳���ʵ�ֶ���################################################
###�б�ͷ���ǡ���ͷ�����б�β���ǡ���β��
class SQuene():
    def __init__(self):
        self._elems = []
    
    def is_empty(self):
        return self._elems == []
    
    def getHead(self):  #�鿴����Ԫ��
        if self._elems == []:
            return str('�ն���')
        return self._elems[0]
    
    def enquene(self, e):  ##���ӣ�ʱ�临�Ӷ�O(1)
        self._elems.append(e)
        
    def dequene(self):  ##���ӣ�ʱ�临�Ӷ�O(n)
        if self._elems == []:
            return str('�ն���')
        return self._elems.pop(0)

        
        
        
###��β�ڵ������ʵ�ֶ���################################################
###O(1)�����������
###����ͷ��Ϊ����ͷ�����б�β��Ϊ����β�����Ӷ�β�����Ӷ�ͷ��
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
    
    def getHead(self):  #�鿴����Ԫ��
        if self.pHead is None:
            return str('�ն���')
        pTemp = self.pHead
        pTemp = pTemp.next
        pTemp = None
        return pTemp.elem
    
    def enquene(self, elem):  #����
        if self.pHead is None:
            self.pHead = LNode(elem, None)  #��ʱpHead=None
            self.pRear = pHead
        else:
            self.pRear.next = LNode(elem, None)
            self.pRear = self.pRear.next

        
    def dequene(self):  #���ӣ�ɾ����ͷԪ�أ�
        if self.pHead is None:
            return str('�ն���')
        e = self.pHead.elem
        self.pHead = self.pHead.next
        return e
        
        
        
        
        
###����ջʵ�ֶ���#############################################################		
###ֱ��listʵ�ְ�
class Stack_to_Quene():
    def __init__(self):
        self.stack_1 = []  #���ջ
        self.stack_2 = []  #����ջ
        
    def is_empty(self):  
        if self.stack_1 == [] and self.stack_2 == []:
            return True
        else:
            return False
            
    def getHead(self):  #�鿴����Ԫ��
        if self.stack_1 == [] and self.stack_2 == []:
            return str('�ն���')
        if self.stack_2 != []:
            return self.stack_2[-1]
        else:
            while len(self.stack_1) > 0:
                self.stack_2.append(self.stack_1.pop())
            return self.stack_2[-1]
            
    def enquene(self, node):  #����
        self.stack_1.append(node)  # �б�ķ�������һ��Ԫ����ӵ�ĩβ

    def dequene(self):  #����
        if len(self.stack_2) > 0:
            return self.stack_2.pop()
        while len(self.stack_1) > 0:
            self.stack_2.append(self.stack_1.pop())
        if len(self.stack_2) == 0:
            return str('�ն���')
        return self.stack_2.pop()  # �б�ķ�����ɾ���б����һ��Ԫ�أ�������Ԫ�ط���



###stack�����
class Stack_to_Quene_2():
    def __init__(self):
        self.stack_1 = SStack()
        self.stack_2 = SStack()
        
    def is_empty(self):  
        if self.stack_1.is_empty() and self.stack_2.is_empty():
            return True
        else:
            return False
            
    def getHead(self):  #�鿴����Ԫ��
        if self.stack_1.is_empty() and self.stack_2.is_empty():
            return str('�ն���')
        if not self.stack_2.is_empty():
            return self.stack_2.getHead()
        else:
            while not self.stack_1.is_empty():
                self.stack_2.push(self.stack_1.pop())
            return self.stack_2.getHead()

    def enquene(self, node):
        self.stack_1.push(node)  # �б�ķ�������һ��Ԫ����ӵ�ĩβ

    def dequene(self):
        if not self.stack_2.is_empty():
            return self.stack_2.pop()
        while not self.stack_1.is_empty():
            self.stack_2.push(self.stack_1.pop())
        if self.stack_2.is_empty():
            return str('�ն���')
        return self.stack_2.pop()  # �б�ķ�����ɾ���б����һ��Ԫ�أ�������Ԫ�ط���


        
###���ȶ���################################################################
'''
���ȶ��е����ʣ��κ�ʱ����ʻ��ߵ����ģ����Ƕ����������ȼ���ߵ�Ԫ��
'''
###���Ա����ȶ��У�
class PrioQue():
    def __init__(self, e_list = []):
        self._elems = list(e_list)
        self._elems.sort(reverse = True)   #��С����Ϊ������
        
    def enquene(self, e):  #�Ҳܣ����ֲ��ҵõ�����λ�ò��Ϳ�������
        inser_index = len(self._elems) - 1
        while inser_index >= 0:
            if self._elems[inser_index] < e:
                inser_index -= 1
            else:
                break
        self._elems.insert(inser_index + 1, e)
    
    def is_empty(self):
        return self._elems == []
    
    def getHead(self):  #�鿴����Ԫ��
        if self._elems == []:
            return str('�ն���')
        return self._elems[-1]
        
    def dequene(self):  ##���ӣ�ʱ�临�Ӷ�O(1)
        if self._elems == []:
            return str('�ն���')
        return self._elems.pop()
        
###��ʵ�����ȶ��У�
