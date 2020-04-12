##############################################
##大数据查找排序算法!!!
##位图算法(适用于非负整数)
class bitMap_Table():
    def __init__(self, MAX_VAL):
        self.max_val = MAX_VAL
        
        size = self.max_val // 31  #下位取整
        if self.max_val % 31 > 0:  #加这句用于上位取整
            size += 1
        
        self.bitMap_array = [0] * size
        
    def insert(self, array):
        for num in array:
            elem_idx = num // 31
            bit_idx = num % 31
            elem = self.bitMap_array[elem_idx]
            self.bitMap_array[elem_idx] = elem | (1<<bit_idx)  
            ##特别注意，list的每个元素用第0~30位存数据，第31位为0
            ##因为python是有符号的32为，最后1位记录正负

    def search(self, tar):
        elem_idx = tar // 31
        bit_idx = tar % 31
        if self.bitMap_array[elem_idx] & (1<<bit_idx):
            return True
        else:
            return False
            
    def sort(self):
        sort_array = []
        for i in range(0, self.max_val):        
            elem_idx = i // 31
            bit_idx = i % 31
            if self.bitMap_array[elem_idx] & (1<<bit_idx):
                sort_array.append(i)
        return sort_array
            
                    
array = [5,2,1,3,6,4,32,54,12,22,67,34]

bitMap = bitMap_Table(100)
bitMap.insert(array)
print(bitMap.search(3))
print(bitMap.sort())



##############################################
##模拟轮盘（自己写的，不是剑指的题）
import random
class Roulette():
    def __init__(self, item_prob_dict):
        assert (sum(item_prob_dict.values()) == 1), print('你的概率加起来不等于1啊！！')
        self.Roulette = [[], []]
        self.Roulette = self.create_Roulette(item_prob_dict)
    
    def create_Roulette(self, item_prob_dict):
        #构建轮盘
        item_list = list(item_prob_dict.keys())
        prob_list = list(item_prob_dict.values())

        n_item = len(item_list)
        Roulette = [[0] * n_item,  item_list]
        for i in range(0, n_item):
            Roulette[0][i] = Roulette[0][i - 1] + prob_list[i]
        return Roulette
    
    def Roulette_output(self):
        #轮盘输出
        x = random.uniform(0, 1)
        for i in range(0, len(self.Roulette[0])):
            if self.Roulette[0][i] >= x:
                break
        return self.Roulette[1][i]

if __name__ == '__main__':
    item_prob_dict = {'哇':0.5, 
                      '联盟':0.8, }
    
    rouletee = Roulette(item_prob_dict)
    rouletee.Roulette_output()