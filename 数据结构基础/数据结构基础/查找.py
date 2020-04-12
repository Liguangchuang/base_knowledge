'''
查找表：同一类型数据元素（记录）的集合；可以理解为一张Excel表，或者一个数组
关键字：数据元素（记录）中某个数据项的值；可以理解为Excel表中“特定列的特定值”，或数组中“特定值”
    主关键字：一个关键字唯一标记一条记录
    次关键字：一个关键字会标记多条记录

'''
'''
    “LTableSearch”通过关键字与目标值进行多次比较，返回关键字的“索引”，操作的复杂度为O(n)或O(logn),后续还需“根据索引”在
查找表查找“关键字对应的数据”，知道索引后的查找复杂度为O(1)
'''

#顺序查找
def order_search(array, key):
    for index in range(0, len(array)-1):
        if array[index] == key:
            return index
    return False

#顺序查找改进版！原来的循环中需要进行2次比较，现在比较一次就可以了
def order_search_improve(array, key):
    if array[0] == key:
        return 0
        
    array[0] = key  #设置“哨兵”
    index = len(array)-1    #从尾部开始查找
    while (array[index] != key):
        index -= 1
        
    if index==0:
        return False
    return index
###算法分析：最好情况是在第一个位置就找到了，此为O(1)；最坏情况在最后一个位置才找到，此为O(n)；
###所以平均查找次数为(n+1)/2。最终时间复杂度为O(n)
    
	
	

###二分查找：复杂度：O(log(n))
def binary_search(array, key):
    low_index = 0
    high_index = len(array) - 1
    
    while(low_index <= high_index):                ##为了方便与“插入查找”做对比，可化简如下：
        mid_index = (low_index + high_index) // 2  ##mid_index = low_index + (high_index - low_index) * (1/2)
        
        if key < array[mid_index]:
            high_index = mid_index - 1
        elif key > array[mid_index]:
            low_index = mid_index + 1
        elif key == array[mid_index]:
            return mid_index
    return False
    
    
###插值查找：复杂度：O(log(n))，与二分查找基本一样~
#二分查找：“折点”永远是中间值。
#插值查找是：目标值越小，“折点”的位置越靠前，目标值越大，“折点”的位置要越往后！
def inser_search(array, key):
    low_index = 0
    high_index = len(array) - 1
    
    while(low_index < high_index):  #注意这里是“<” ；不能是“<=”
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
##有点麻烦，就不写成字典的形式了吧！！！
###搜索二叉树
class BinTree_search(self):
    def __init__(self):
        pRoot = None

    ##插入	
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
        
    ##搜索（返回搜索到子树）
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
        
    ##删除：
        
        
        
########################################################################################
'''
索引在这里直接理解为下标：索引——下标
如果关键字就是存储数据的位置，就可以直接O(1)查找数据；但是关键字的范围可能太大，不合适作为下标
于是有了hash表的思想：通过一个hash函数将“关键字映射为下标”，这样在查找的时候就可以直接“通过关键字计算出“下标”
hash表有两部操作：
    关键字存入hash表：
    hash表内查找关键字的“下标”：
    
两个关键技术：1、hash函数的设置：这里使用“求余法”
              2、冲突消解：开位置法和探查方式  
hash函数：把任意长度的输入，通过映射，变换成固定长度的输出
'''
class HashTable_search():  
    def __init__(self, size):
        self.h_array = [None for i in range(0, size)]
        self.count = size
        self.m = self.count  #这里直接用表长来“取余”；实际一般取 “<=表长”

    def hash_func(self, key):
        address = key % self.m
        return address   ##hash函数为：求余法
        
    ##将关键字插入到hash表内
    def insert_hash(self, key):  #将关键字插入hash表
        address = self.hash_func(key)  # 求key在散列表的位置
        while self.h_array[address]:  #该位置已经存在值，表示“存在冲突”
            address = (address + 1) % self.m   ##用“开放位置法”寻找位置；每次向右移动一格查找
        self.h_array[address] = key  #找到位置后，保存到该位置
       
    ##基于哈希表的搜索
    def search_hash(self, key):
        start = address = self.hash_func(key)
        while self.h_array[address] != key:  #如果现在的位置有值，表示冲突
            address = (address + 1) % self.m 
            if (self.h_array[address] is None) or (address == start):  #没有找到，或者找回了原点
                return False
        return True  
    

array = [12, 67, 56, 16, 25, 37, 22, 29, 15, 47, 48, 34]
hashTable_search = HashTable_search(12)

#建立hash表（将关键字插入到hash表内）
for item in array:  
    hashTable_search.inser_hash(item)

#显示关键字在h_array中的位置
for item in hashTable_search.h_array:
    print((item, hashTable_search.h_array.address(item)), end=" ")
hashTable_search.h_array
hashTable_search.search_hash(15)  
hashTable_search.search_hash(90)

