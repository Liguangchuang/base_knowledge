'''
���ұ�ͬһ��������Ԫ�أ���¼���ļ��ϣ��������Ϊһ��Excel������һ������
�ؼ��֣�����Ԫ�أ���¼����ĳ���������ֵ���������ΪExcel���С��ض��е��ض�ֵ�����������С��ض�ֵ��
    ���ؼ��֣�һ���ؼ���Ψһ���һ����¼
    �ιؼ��֣�һ���ؼ��ֻ��Ƕ�����¼

'''
'''
    ��LTableSearch��ͨ���ؼ�����Ŀ��ֵ���ж�αȽϣ����عؼ��ֵġ��������������ĸ��Ӷ�ΪO(n)��O(logn),�������衰������������
���ұ���ҡ��ؼ��ֶ�Ӧ�����ݡ���֪��������Ĳ��Ҹ��Ӷ�ΪO(1)
'''

#˳�����
def order_search(array, key):
    for index in range(0, len(array)-1):
        if array[index] == key:
            return index
    return False

#˳����ҸĽ��棡ԭ����ѭ������Ҫ����2�αȽϣ����ڱȽ�һ�ξͿ�����
def order_search_improve(array, key):
    if array[0] == key:
        return 0
        
    array[0] = key  #���á��ڱ���
    index = len(array)-1    #��β����ʼ����
    while (array[index] != key):
        index -= 1
        
    if index==0:
        return False
    return index
###�㷨���������������ڵ�һ��λ�þ��ҵ��ˣ���ΪO(1)�����������һ��λ�ò��ҵ�����ΪO(n)��
###����ƽ�����Ҵ���Ϊ(n+1)/2������ʱ�临�Ӷ�ΪO(n)
    
	
	

###���ֲ��ң����Ӷȣ�O(log(n))
def binary_search(array, key):
    low_index = 0
    high_index = len(array) - 1
    
    while(low_index <= high_index):                ##Ϊ�˷����롰������ҡ����Աȣ��ɻ������£�
        mid_index = (low_index + high_index) // 2  ##mid_index = low_index + (high_index - low_index) * (1/2)
        
        if key < array[mid_index]:
            high_index = mid_index - 1
        elif key > array[mid_index]:
            low_index = mid_index + 1
        elif key == array[mid_index]:
            return mid_index
    return False
    
    
###��ֵ���ң����Ӷȣ�O(log(n))������ֲ��һ���һ��~
#���ֲ��ң����۵㡱��Զ���м�ֵ��
#��ֵ�����ǣ�Ŀ��ֵԽС�����۵㡱��λ��Խ��ǰ��Ŀ��ֵԽ�󣬡��۵㡱��λ��ҪԽ����
def inser_search(array, key):
    low_index = 0
    high_index = len(array) - 1
    
    while(low_index < high_index):  #ע�������ǡ�<�� �������ǡ�<=��
        mid_index = low_index + int((high_index - low_index) * ((key-array[low_index]) / (array[high_index]-array[low_index])))

        if key < array[mid_index]:
            high_index = mid_index - 1
        elif key > array[mid_index]:
            low_index = mid_index + 1
        elif key == array[mid_index]:
            return mid_index
    return False


key = 4
array = [1,2,3,4]

#order_search(array, key)
#order_search_improve(array, key)
#binary_search(array, key)
inser_search(array, key)





#########################################################################################
##�е��鷳���Ͳ�д���ֵ����ʽ�˰ɣ�����
###����������
class BinTree_search(self):
    def __init__(self):
        pRoot = None

    ##����	
    def insert(self, key):
        bT = self.pRoot
        if bT is None:
            self.pRoot = BinTNode(key)
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
        
    ##����������������������
    def search(self, key):  
        bT = self.pRoot
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
        
        
        
########################################################################################
'''
����������ֱ�����Ϊ�±꣺���������±�
����ؼ��־��Ǵ洢���ݵ�λ�ã��Ϳ���ֱ��O(1)�������ݣ����ǹؼ��ֵķ�Χ����̫�󣬲�������Ϊ�±�
��������hash���˼�룺ͨ��һ��hash���������ؼ���ӳ��Ϊ�±ꡱ�������ڲ��ҵ�ʱ��Ϳ���ֱ�ӡ�ͨ���ؼ��ּ�������±ꡱ
hash��������������
    �ؼ��ִ���hash��
    hash���ڲ��ҹؼ��ֵġ��±ꡱ��
    
�����ؼ�������1��hash���������ã�����ʹ�á����෨��
              2����ͻ���⣺��λ�÷���̽�鷽ʽ  
hash�����������ⳤ�ȵ����룬ͨ��ӳ�䣬�任�ɹ̶����ȵ����
'''
class HashTable_search():  
    def __init__(self, size):
        self.h_array = [None for i in range(0, size)]
        self.count = size
        self.m = self.count  #����ֱ���ñ�����ȡ�ࡱ��ʵ��һ��ȡ ��<=����

    def hash_func(self, key):
        address = key % self.m
        return address   ##hash����Ϊ�����෨
        
    ##���ؼ��ֲ��뵽hash����
    def insert_hash(self, key):  #���ؼ��ֲ���hash��
        address = self.hash_func(key)  # ��key��ɢ�б��λ��
        while self.h_array[address]:  #��λ���Ѿ�����ֵ����ʾ�����ڳ�ͻ��
            address = (address + 1) % self.m   ##�á�����λ�÷���Ѱ��λ�ã�ÿ�������ƶ�һ�����
        self.h_array[address] = key  #�ҵ�λ�ú󣬱��浽��λ��
       
    ##���ڹ�ϣ�������
    def search_hash(self, key):
        start = address = self.hash_func(key)
        while self.h_array[address] != key:  #������ڵ�λ����ֵ����ʾ��ͻ
            address = (address + 1) % self.m 
            if (self.h_array[address] is None) or (address == start):  #û���ҵ��������һ���ԭ��
                return False
        return True  
    

array = [12, 67, 56, 16, 25, 37, 22, 29, 15, 47, 48, 34]
hashTable_search = HashTable_search(12)

#����hash�����ؼ��ֲ��뵽hash���ڣ�
for item in array:  
    hashTable_search.inser_hash(item)

#��ʾ�ؼ�����h_array�е�λ��
for item in hashTable_search.h_array:
    print((item, hashTable_search.h_array.address(item)), end=" ")
hashTable_search.h_array
hashTable_search.search_hash(15)  
hashTable_search.search_hash(90)

