'''
ð������
    ��һ��forѭ����round��ÿһ�ְ��������ŵ����ұߣ�Ȼ��ѾͲ���Ҫ�ٹ��ǲ������ݣ�2����ð��1�֣�3����ð��2�֣�
�Ϳ�����ȫ�ź������һ��������Ҫ��ð�ݣ������Ҫ��roundΪn-1      
    �ڶ���forѭ����cu_index����ǰλ�ú�ǰһ��λ�ñȽϣ�ÿһ�ֶ�����Ҫ�ٹ��ұ�������Ĳ��֣�ֻ��ʣ������Ƚϱ�ɣ�
cu_index���ƶ��±�Ϊ��1��n-round 

    �ܶ��㷨û�С�return array������Ϊ��ʹ�ǵ����Ӻ�����Ҳ��ֱ�Ӹı�ԭ���ı�����list������Ż�ı䣬����ֵ���ı䣩��
Ҳ����֮ǰ��array�ı��ˣ���
'''

###Ĭ�������Ǵӡ�С����

#ð������
def bubble_sort(array):  
    for round in range(len(array)-1):  #2������1�֣�3������2��
        found = False  #Ĭ�ϲ���������
        for cu_index in range(0, len(array)-1-round):  #cu_index����Ҫ�ƶ����Ѿ��ź�������� 
            if array[cu_index] > array[cu_index+1]:
                array[cu_index], array[cu_index+1] = array[cu_index+1], array[cu_index]
                found = True
        if found == False:  #�������������Ҳ������������Ѿ��ź��򣬾�ֱ���˳�ѭ��
            break
    
    
	
	
##ѡ������
#����ԭ��ÿһ�δӴ����������Ԫ����ѡ����С��һ��Ԫ�أ���������е���ʼλ�ã�ֱ��ȫ�������������Ԫ������
def select_sort(array):
    for round_index in range(0, len(array)-1):
        min_index = round_index
        for cu_index in range(round_index, len(array)):
            if array[cu_index] < array[min_index]:
                min_index = cu_index
        if round_index != min_index:
            array[round_index], array[min_index] = array[min_index], array[round_index]



###��������
def insert_sort(array):
    for round_index in range(1, len(array)):
        round_value = array[round_index]
        inser_index = round_index
        while inser_index > 0 and array[inser_index-1] > round_value:  ##��ǰ���ֵ���������ֵ�ֵ��ʱ
            array[inser_index] = array[inser_index-1]  ##Ԫ����������ƶ���
            inser_index -= 1
        array[inser_index] = round_value  ##round_value�Ĳ���
    



https://www.cnblogs.com/MOBIN/p/4681369.html
###���������ص㣩
def partition(array, left_index, right_index):
    base_value = array[left_index]  ###ָ��λ�ö��ڿ��ŵ������ǳ���Ҫ����������ѡ������ߵ�λ����Ϊָ��λ�ã�
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
    return left_index  ##��������return��
    
def quick_sort(array, left_index, right_index):
    if left_index < right_index:
        mid_index = partition(array, left_index, right_index)
        quick_sort(array, left_index, mid_index-1)  ##part_index��Ԫ�ز���Ҫ����������
        quick_sort(array, mid_index+1, right_index)


    
    
https://www.cnblogs.com/chengxiao/p/6194356.html
###���������ص㣩
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
    return merge_array  ##��������return��

def merge_sort(self, array):
    if len(array) <= 1:
        return array  #�ݹ�Ľ���
    mid_index = len(array) // 2  #//��ʾȡ����
    left_array = merge_sort(array[ :mid_index])
    right_array = merge_sort(array[mid_index: ])
    return merge(left_array, right_array)  ##�鲢��return

    
    
    
def heap_sort()




	
	
array = [5,2,45,6,8,2,1]
    
#bubble_sort(array)
#elect_sort(array)
#insert_sort(array)
#quick_sort(array,0, len(array)-1)
merge_sort(array)  ##����鲢�����з���ֵ
