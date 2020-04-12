'''
目标串target： ahdksicdpsdxsd
模式串pattern：dksi
'''

##计算机，我真想说，草拟大爷，想法简单得要死，但你要写出来很麻烦，而且各种边界条件非常容易出错，
##总之，很简单的想法，给计算机搞就很麻烦！！！
###朴素串匹配算法：

def naive_matching(target, pattern):
    len_pat, len_tar = len(pattern), len(target)
    index_pat, index_tar = 0, 0
    
    while (index_pat <= len_pat-1) and (index_tar <= len_tar-1):
        if pattern[index_pat] == target[index_tar]:
            index_pat += 1
            index_tar += 1
        else:
            index_tar = index_tar - index_pat + 1
            index_pat = 0
    if index_pat == len_pat:
        tar_start_idx = (index_tar - len_pat)
        return tar_start_idx
    else:
        return -1
    
    
def naive_matching_1(target, pattern):
    def is_same(sub_tar, pattern):
        for idx in range(len(sub_tar)):
            if sub_tar[idx] != pattern[idx]:
                return False
        return True
    
    tar_idx = 0
    while tar_idx <= len(target)-len(pattern):
        sub_tar = target[tar_idx: tar_idx+len(pattern)]
        if is_same(sub_tar, pattern):
            return tar_idx
        tar_idx += 1
    return -1
        
target = 'ahdksicdpsdxsd'
pattern = 'dksi'
print(naive_matching_1(target, pattern))
