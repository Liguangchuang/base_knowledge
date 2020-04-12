'''
冒泡排序：
    第一个for循环：round，每一轮把最大的数排到最右边，然后把就不需要再管那部分数据；2个数冒泡1轮，3个数冒泡2轮，
就可以完全排好序；最后一个数不需要在冒泡，因此需要的round为n-1      
    第二个for循环：cu_index，当前位置和前一个位置比较，每一轮都不需要再管右边已排序的部分，只对剩余的数比较便可，
cu_index的移动下标为从1到n-round 

    很多算法没有“return array”，因为即使是调用子函数，也会直接改变原来的变量（list、链表才会改变，单个值不改变），
也就是之前的array改变了！！
'''

###默认排序是从“小到大”

#冒泡排序
def bubble_sort(array):  
    for round in range(len(array)-1):  #2个数排1轮，3个数排2轮
        found = False  #默认不存在逆序
        for cu_index in range(0, len(array)-1-round):  #cu_index不需要移动到已经排好序的区域 
            if array[cu_index] > array[cu_index+1]:
                array[cu_index], array[cu_index+1] = array[cu_index+1], array[cu_index]
                found = True
        if found == False:  #如果不存在逆序，也就是这个数组已经排好序，就直接退出循环
            break
    
    
	
	
##选择排序
#工作原理：每一次从待排序的数据元素中选出最小的一个元素，存放在序列的起始位置，直到全部待排序的数据元素排完
def select_sort(array):
    for round_index in range(0, len(array)-1):
        min_index = round_index
        for cu_index in range(round_index, len(array)):
            if array[cu_index] < array[min_index]:
                min_index = cu_index
        if round_index != min_index:
            array[round_index], array[min_index] = array[min_index], array[round_index]



###插入排序
def insert_sort(array):
    for round_index in range(1, len(array)):
        round_value = array[round_index]
        inser_index = round_index
        while inser_index > 0 and array[inser_index-1] > round_value:  ##当前面的值比现在这轮的值大时
            array[inser_index] = array[inser_index-1]  ##元素逐个向右移动，
            inser_index -= 1
        array[inser_index] = round_value  ##round_value的插入
    



https://www.cnblogs.com/MOBIN/p/4681369.html
###快速排序（重点）
def partition(array, left_index, right_index):
    base_value = array[left_index]  ###指定位置对于快排的质量非常重要！！（现在选定最左边的位置作为指定位置）
    while left_index < right_index:
        while (left_index < right_index) and (array[right_index] >= base_value):
            right_index -= 1
        if left_index < right_index:
            array[left_index] = array[right_index]
        while (left_index < right_index) and array[left_index] <= base_value:
            left_index += 1
        if left_index < right_index:
            array[right_index] = array[left_index]
    array[left_index] = base_value
    return left_index  ##这里是有return的
    
def quick_sort(array, left_index, right_index):
    if left_index < right_index:
        mid_index = partition(array, left_index, right_index)
        quick_sort(array, left_index, mid_index-1)  ##part_index的元素不需要再纳入排序
        quick_sort(array, mid_index+1, right_index)


    
    
https://www.cnblogs.com/chengxiao/p/6194356.html
###并归排序（重点）
def merge(left_array, right_array):
    l_arr_index = 0
    r_arr_index = 0
    merge_array = []
    while l_arr_index < len(left_array) and r_arr_index < len(right_array):
        if left_array[l_arr_index] <= right_array[r_arr_index]:
            merge_array.append(left_array[l_arr_index])
            l_arr_index += 1
        else:
            merge_array.append(right_array[r_arr_index])
            r_arr_index += 1
    merge_array += left_array[l_arr_index: ]
    merge_array += right_array[r_arr_index: ]
    return merge_array  ##这里是有return的

def merge_sort(self, array):
    if len(array) <= 1:
        return array  #递归的结束
    mid_index = len(array) // 2  #//表示取整数
    left_array = merge_sort(array[ :mid_index])
    right_array = merge_sort(array[mid_index: ])
    return merge(left_array, right_array)  ##归并有return

    
    
    
def heap_sort()




	
	
array = [5,2,45,6,8,2,1]
    
#bubble_sort(array)
#elect_sort(array)
#insert_sort(array)
#quick_sort(array,0, len(array)-1)
merge_sort(array)  ##这个归并排序有返回值
