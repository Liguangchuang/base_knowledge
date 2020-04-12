###############################################################################
'''给定4个点，判断是否能组成正方形'''
def calcu_distance(p1, p2):
    return (p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1])
    
def is_square(p1, p2, p3, p4): 
    distance_list = []
    distance_list.append(calcu_distance(p1, p2))
    distance_list.append(calcu_distance(p1, p3))
    distance_list.append(calcu_distance(p1, p4))
    distance_list.append(calcu_distance(p2, p3))
    distance_list.append(calcu_distance(p2, p4))
    distance_list.append(calcu_distance(p3, p4))
    
    len1 = distance_list[0]
    for distan in distance_list:
        if distan != len1:
            len2 = distan
            breakz
    
    num1 = 0
    num2 = 0
    for distan in distance_list:
        if distan == len1:
            num1 += 1
        elif distan == len2:
            num2 += 1
    
    if (num1 == 4 and num2 == 2) or (num1 == 2 and num2 == 4):
        return True
    else:
        return False
    
p1 = [10, 0]
p2 = [10, 1]
p3 = [11, 0]
p4 = [11, 1]

print(is_square(p1, p2, p3, p4))




    

    
    
###求所有的完数（完数：所有的因子的和刚好等于这个数本身）######################################################
def com_baiyi_getPerfectNumbers(n):
	arr = []
	for m in range(1, n+1):
		arr_yinzi = ji_suan_yin_zi(m)
		if sum(arr_yinzi) == m:
			arr.append(m)
	return arr
	

def ji_suan_yin_zi(m):
	arr = []
	
	for i in range(1, m):
		if m % i == 0:
			arr.append(i)
	return arr


print(com_baiyi_getPerfectNumbers(40))


###求两个数的最大公约数###############################################################################
def hfc(x, y):
    if x > y:
        smaller = y
    else:
        smaller = x
    
    
    for i in range(1, smaller+1):
        if (x % i == 0) and (y % i == 0):
            hcf_ = i
    
    return hcf_

arr_tmp = input().split(' ')  ##特别重要
arr = []
for e in arr_tmp:
    arr.append(int(e))
x = arr[0]
y = arr[1]

hfc(x, y)


###快速排序###############################################################################
def partition(arr, l_idx, r_idx):
    base_value = arr[l_idx]  ###指定位置对于快排的质量非常重要！！（现在选定最左边的位置作为指定位置）
    while l_idx < r_idx:
        while (l_idx < r_idx) and (arr[r_idx] >= base_value):
            r_idx -= 1
        arr[l_idx] = arr[r_idx]
        
        while (l_idx < r_idx) and arr[l_idx] <= base_value:
            l_idx += 1
        arr[r_idx] = arr[l_idx]
        
    arr[l_idx] = base_value
    return l_idx  ##这里是有return的

def quick_sort(arr, l_idx, r_idx):
    if l_idx < r_idx:
        mid_index = partition(arr, l_idx, r_idx)
        self.quick_sort(arr, l_idx, mid_index-1)  ##part_index的元素不需要再纳入排序
        self.quick_sort(arr, mid_index+1, r_idx)



###找出最小k个数##########################################################################




###稀疏矩阵的乘法##########################################################################
def dense_to_sparse(dense_mat):
    size = [len(dense_mat), len(dense_mat[0])]
    spare_mat = []
    
    for r in range(0, len(dense_mat)):
        for c in range(0, len(dense_mat[0])):
            if dense_mat[r][c] != 0:
                spare_mat.append((r, c, dense_mat[r][c]))  
                
    return spare_mat, size

def sparse_to_dense(spare_mat, size):
    dense_mat = [[0] * size[1] for row in range(size[0])]
    
    for tuple_ in spare_mat:
        dense_mat[tuple_[0]][tuple_[1]] = tuple_[2]

    return dense_mat


def matrix_multiply(A, B):
    res = [[0 for c in range(0, len(B[0]))] for r in range(0, len(A))] #先构造一个符合size的全0矩阵
    
    for Ar in range(0, len(A)):
        for Ac in range(0, len(A[0])):
            if A[Ar][Ac] != 0: #对于是0的元素，就不需要修改res了
                for Bc in range(0, len(B[0])):
                    if B[Ac][Bc] != 0: #对于是0的元素，就不需要修改res了
                        res[Ar][Bc] += A[Ar][Ac] * B[Ac][Bc]
    return res


if __name__ == '__main__':
    A = [[1,0,0],[-1,0,3]]
    B = [[7,0,0],[0,0,0],[0,0,1]]
    result = SparseMatrixMultiply(A, B)
    print(result)
    



##############################################################################
##数组的长度为101，值域为[1,100]，有且仅有一个数字重复，找出该数字，要求空间复杂度为O(1)，时间复杂度为O(n):
def solu(arr):
    for idx in range(0, len(arr)):
        if arr[idx] in (arr[0: idx] + arr[idx+1: ]):
            return arr[idx]
            
            
def solu(arr):
    sum_ = 0
    for i in range(0, len(arr)):
        sum_ += i
    res = sum_ - (n(n-1)/2)
    
    return res
    
        
    

##############################################################################
##递归实现阶层函数        
def fac(n):
    if n == 1:
        return 1
    elif n <= 0:
        return 0

    return n * fac(n-1)







