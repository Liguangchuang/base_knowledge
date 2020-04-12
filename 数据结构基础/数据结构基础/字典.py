class Dict_elem():
    def __init__(self, key, value):
        self.key = key
        self.value = value
        
class BinTNode():
    def __init__(self, x, left=None, right=None):
        self.data = x
        self.left = left
        self.right = right
'''
�ֵ�Ļ���������
    �Ƿ�Ϊ���ա�
    ����
    ����
    ɾ��
'''	

##���Ա��ֵ�
class Dict_LTable():  
    def __init__(self):
        self.array = []

    def __binary_search(self, array, key):
        low_index = 0
        high_index = len(array) - 1
        
        while(low_index <= high_index):                
            mid_index = (low_index + high_index) // 2  
            
            if key < array[mid_index].key:
                high_index = mid_index - 1
            elif key > array[mid_index].key:
                low_index = mid_index + 1
            elif key == array[mid_index].key:
                return mid_index
        return False
        
        
        
    def is_empty(self):
        return self.array == []
        
    def insert(self, key, value):
        low_index = 0
        high_index = len(array) - 1

        if array == []:
            self.array.insert(0, Dict_elem(key, value))
            return 
        #���жϱ߽�
        if key < array[low_index].key:
            self.array.insert(low_index, Dict_elem(key, value))
            return 
        elif key > array[low_index].key:
            self.array.insert(high_index, Dict_elem(key, value))
            return 
        elif key == array[low_index].key:
            array[low_index].value = value
            return 
        elif key == array[high_index].key:
            array[high_index].value = value
            return
        #�ж��м�
        while(low_index < high_index):         
            if (high_index - low_index) == 1:
                self.array.insert(high_index, Dict_elem(key, value))
                return 
                
            mid_index = (low_index + high_index) // 2  
            mid_dict_elem = array[mid_index]
            if key < mid_dict_elem.key:
                high_index = mid_index 
            elif key > mid_dict_elem.key:
                low_index = mid_index 
            elif key == mid_dict_elem.key:
                mid_dict_elem.value = value
                return 


    def search(self, key):
        search_index = self.__binary_search(self.array, key)
        if search_index is False:
            return False
        sear_dict_elem = self.array[search_index]
        return sear_dict_elem.value
        
    def delete(self, key):
        delete_index = self.__binary_search(self.array, key)
        if delete_index is False:
            return False
        del self.array[delete_index]

        
        
        
##�����������ֵ�		
class Dict_BinSortTree():  ##ֱ�ӷ��عؼ��ֶ�Ӧ�ġ�ֵ��������Ҫ��������ֱ�ӷ��صľ������ݣ���
    def __init__(self):
        self.pRoot = None
        
    def is_empty(self):
        return self.pRoot is None 
        
    
    def insert(self, key, value):
        bT = self.pRoot
        if bT is None:
            self.pRoot = BinTNode(Dict_elem(key, value))
            return 
        while True:
            dict_elem = bT.data
            if key < dict_elem.key:
                if bT.left is None:
                    bT.left = BinTNode(Dict_elem(key, value))
                    return 
                bT = bT.left
            elif key > dict_elem.key:
                if bT.right is None:
                    bT.right = BinTNode(Dict_elem(key, value))
                    return 
                bT = bT.right
            else:
                dict_elem.value = value
                return 
        
        
    def search(self, key):  
        bT = self.pRoot
        while bT:
            dict_elem = bT.data  ##���Ľڵ�Ϊ��key-vaule������dict_���ֵ䣬�С�key-vakue����
            if key < dict_elem.key:
                bT = bT.left
            elif key > dict_elem.key:
                bT = bT.right
            elif key == dict_elem.key:
                return dict_elem.value  ##ֱ�ӷ����ֵ��value
        return False
        
    
    def delete(self, key):  #��ӵĲ���������
        
    
    
        
##��ϣ���ֵ�
class Dict_HashTable():
    def __init__(self):
        INIT_SIZE = 8
        self.count_all = INIT_SIZE
        self.count_act = 0
        self.mod = self.count_all  #����ֱ���ñ�����ȡ�ࡱ��ʵ��һ��ȡ ��<=����
        self.h_array = [None for i in range(0, count_all)]
        
    def __hash_func(self, key):
        return key % self.mod   ##hash����Ϊ�����෨
        
    def __search_hash_addr(self, key):
        start = address = self.__hash_func(key)
        while self.h_array[address] != key:  #������ڵ�λ����ֵ����ʾ��ͻ
            address = (address + 1) % self.mod 
            if (self.h_array[address] is None) or (address == start):  #û���ҵ��������һ���ԭ��
                return False
        return address  
        
    def __insert_hash_addr(self, key):
        address = self.__hash_func(key)  
        while self.h_array[address]:  
            address = (address + 1) % self.mod   
        return address
        
    def __is_extend_space(self):
        load_factor = self.count_act / self.count_all
        
        if load_factor > 0.65:
            self.count_all = 2 * self.count_all
            self.h_array = [None for i in range(0, count_all)]
            for item in h_array:
                self.insert(self, item.key, item.value)
                self.count_act -= 1
    
    def __is_reduce_space(self):
        load_factor = self.count_act / self.count_all
        
        if load_factor < 0.1:
            self.count_all = (1/2) * self.count_all
            self.h_array = [None for i in range(0, count_all)]
            for item in h_array:
                self.insert(self, item.key, item.value)
                self.count_act += 1
        
        
    
    def is_empty(self):   
        for i in range(0, len(h_array)):
            if h_array[i]:
                return False 
            return True  
        
    def insert(self, key, value):
        self.__is_extend_space()
        
        insert_addr = self.__insert_hash_addr(key)
        self.h_array[insert_addr] = Dict_elem(key, value)
        self.count_act += 1
        
        
    def search(self, key):  
        search_addr = self.__search_hash_addr(key)
        if search_addr is False:
            return False
        dict_elem = self.h_array[search_addr]
        return dict_elem.value
    
    
    def delete(self, key):
        self.__is_reduce_space()
        
        delete_addr = self.__search_hash_addr(key)
        if delete_addr is False:
            return False
        self.h_array[delete_addr] = None
        self.count_act -= 1
