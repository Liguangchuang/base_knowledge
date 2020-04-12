import os
import sys 
from time import time
import random
import json
import numpy as np
import pandas as pd
import pickle
import codecs

import jieba
import nltk
import lightgbm as lgb
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer


import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec

pd.set_option('display.max_columns', 100) #最大行数与最大列数   
pd.set_option('display.max_rows', 100) 
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"  #'last', 'None'

in_path = '../workCode/'   
out_path = '../workCode/'

in_path = 'E:/workCode/'   #也可以写成绝对路径（注意不能有中文命名的文件夹）
out_path = 'E:/workCode/'




########################################################################################################################################
###数据读入，数据输出###########################
#pandas
df_yaShiLanDai = pd.read_csv(in_path + 'yaShiLanDai.csv')  #如果报编码错误，可以加encoding='ISO-8859-1'
weiBo_outPut.to_csv(out_path + 'weiBo_outPut.csv', index =False)

#注意：如果加载不定长的文本str，要在最顶层补加一行，而且长度要最大；但pandas垃圾，直接open就行了


#pandas读入所有文件
for f_name in os.listdir(input_path):
    if f_name.endswith(".csv"):
        df = pd.read_csv(input_path+f_name)
        print(df.head())

#pickle
def pickle_store(df_test,filename):  #存储
    with open(filename, 'w') as fw:
        pickle.dump(df_test, fw)
    
def pickle_load(filename):  #加载
    fr = open(filename, 'r')
    return pickle.load(fr)



#原生python           
with open(file='test.txt', mode='rt', encoding='utf-8') as f:  #“位置”参数必须放在“关键字”参数的前面；
    rawText = f.read()  #读入                                  #open(file='test', 'rt') as f； 报错
    
with open(file='test.txt', mode='wt', encoding='utf-8') as f:
    f.write(train_data_1W)  #写出
                            #注意，encoding：使用rt或wt模型才能使用，编码成某种格式（'gbk', 'utf-8'）


#全部读到列表
with open(file='test.txt', mode='rt', encoding='utf-8') as f:  #“位置”参数必须放在“关键字”参数的前面；
    lines = f.readlines()  #读入，lines以列表的形式保存；每个元素是“一行的str”                                
    
    
#一行一行读入
chi_stopWord = []
with open('chi_stop_words.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line: 
            break
        chi_stopWord.append(line.strip())  #strip表示删除多余的无关字符


#一行一行读入（使用迭代器的读法，更简短）
chi_stopWord = []
with open('chi_stop_words.txt') as f:  #默认是读模式
    for line in f:
        chi_stopWord.append(line.strip())
        
 
#几行几行读入
with open(in_path+'test.raw', 'rt') as f:
    lines = f.readlines()
    while True:
        mul_line = lines[0: 3]
        lines = lines[3: ]
        if len(mul_line) < 3:
            break
        
        ##process mul_line here
        

        
#一行一行输出
chars=['c1', 'c2', 'c3']
tags=['t1', 't2', 't3']
with open('test.txt', 'wt', encoding='utf-8') as f:  
    for (char, tag) in zip(chars, tags):  
        str_ = char + ' ' + tag + '\n'
        f.write(str_)
    
    

#用open()读取文件夹所有的txt文件
for f_name in os.listdir(input_path):
    if f_name.endswith(".txt"):
        with open(file=input_path+f_name, mode='rt', encoding='utf-8') as f:
            a_Text = f.read()[1: ]  #不知道为什么有这个标识\ufeff
            print(a_Text[0: 5])


            
###读取csv数据的神器             
import csv
with open(in_path+'test.csv', 'r', encoding='utf-8') as f:
    f = csv.reader(f, delimiter='\t')  #delimiter指定分隔符，（将文件转化一下便可）
    for row in f:    #row将每一行读成list；还是list的处理快啊！！！
        print(row)
'''
csv, 真的很快吗？, 是的, 很快           ##test.csv
pandas, 为什么这么慢？, 是的, 很慢

['csv', ' 真的很快吗？', ' 是的', ' 很快']      ##print(row)
['pandas', ' 为什么这么慢？', ' 是的', ' 很慢']
'''

##写csv文件
def write_csv_file(path, head, data):  
    with open(path, 'w', newline='') as csv_file:    #newline='' 让他不会自动空一行
        writer = csv.writer(csv_file, dialect='excel')  
        if head is not None:  
            writer.writerow(head)  
        for row in data:  
            writer.writerow(row)  
            
data = [[1,2,3], [4,5,6], [7,8,9], [1,2,3], [7,8,9], [1,2,3]]
write_csv_file('some.csv', None, data)



##df数据########################################################
#实验用的df数据
df = pd.DataFrame({'a':[1,2,3], 'b':[4,5,6], 'c':[7,8,9]}, columns=['a', 'b', 'c'])
df = pd.DataFrame(columns=['a', 'b', 'c'])  #建立一个空的df

#数据分析常用
df.shape  #查看数据维度
df.info() #查看数据信息
df.describe()    #查看数据描述
df.isna().sum()  #查看缺失情况
df.label.unique() #查看标签的种类（重点）

#更改列的顺序
col_names = list(df)
col_names.insert(0, col_names.pop(col_names.index('second')))  #比如将'second'那列放到首列
df = df.reindex(columns=col_names)

#去除空值
filter_data = filter_data.dropna()  #默认去除任何含有空值的“行”
origin_data = origin_data[origin_data.分析字段.notnull()]  #对某一列，选取非空的行

#填充空值
df = df.fillna('nan_value')  #全部填充
df['d'] = df.d.fillna('nan_value')  #只填充"d"列

#数据过滤
df = df[(df['a'] != 1) | (df['b'] != 5)]  #()是一定一定要的，否则报错

#数据替换
filter_data['特征类型'] = filter_data.特征类型.replace({'正面':1, '负面':-1})

#设置索引
df = pd.DataFrame({'k1': [1,2,3,4], 'k2':[4,5,6,7]})  #默认索引是[0,1,2,3...]
#	k1	k2
#0	1	4
#1	2	5
#2	3	6
#3	4	7

df.index=['a', 'b', 'c', 'd']  #将索引设置为['a', 'b', 'c', 'd']
#	k1	k2
#a	1	4
#b	2	5
#c	3	6
#d	4	7

df = pd.DataFrame({'k1': [1,2,3,4], 'k2':[4,5,6,7]})
df = df.set_index('k1')  #将'k1'这列设置为索引
#    k2
#k1	
#1	4
#2	5
#3	6
#4	7

df = pd.DataFrame({'k1': [1,2,3,4], 'k2':[4,5,6,7]})
df = df.reset_index()  #将索引重新设置为range，如果假如drop=True
#	k1	k2
#0	1	4
#1	2	5
#2	3	6
#3	4	7

df = pd.DataFrame({'k1': [1,2,3,4], 'k2':[4,5,6,7]}) 
df = df.reset_index(drop=True)  #将原来那列去掉
#	k2
#0	4
#1	5
#2	6
#3	7


#根据索引取数
df.loc[['a']] #取idx='a'那行
df.loc[['a', 'b']] #取idx='a'和idx='b'这两行
df.loc['a', 'k1'] #取idx='a'且列名为'k1'那个单元格的值


#数据去重：
df = pd.DataFrame({'k': [1, 3, 4, 2], 'j': [1, 3, 4, 2], 'l': [1, 3, 2, 2]})
df.drop_duplicates(["l", 'j'], keep="first").reset_index(drop=True)  #针对某“几列”去重
df.drop_duplicates(keep="first").reset_index(drop=True)  #针对全部数据去重

####删除数据
df = df.drop([1])  #删除index=1的行；列表可以传入多个值
df = df.drop('a', axis=1)  #删除名字是'a'的列

#将某一列，输出为txt，（比直接open()快很多）
origin_data.分析字段.to_csv(in_path + 'fenXiZiDuan.txt', index=False, sep='\a', header=None)  #直接输出便可

#加载txt数据
pd.read_csv(in_path+'fenXiZiDuan.txt', sep='\a', header=None, encoding='gbk')  #指定“分割符”便可，sep='\a'没啥意义，所以选他    

#查看某个字段，各种值得个数
filter_data_.属性.value_counts()


##注意：不要用df数据直接做for循环迭代，非常慢的。
##可以先转化为列表，再用for迭代
domain_list = list(tmp.领域)  #先转化为list，再来迭代
property_list = list(tmp.属性)
dict_domain_property = {}

for i in range(0, len(tmp[0: ])):
    domain = domain_list[i]
    property_ = property_list[i]
    
    if domain not in dict_domain_property.keys():
        dict_domain_property[domain] = set()
    dict_domain_property[domain].add(property_)


    
###numpy#################################################################
#数组合并
#1、类似list的append()效果
a = np.array([[1,3], [4,6]])
b = np.array([1,2,3])
ab = np.append(a, b)  #全部压缩成一列输出，输出[1,2,4,6,1,2,3]





    

###字符串操作############################################################
import re

#抽取夹在中间的str_
def extrac_char(str, start_c, end_c):
    num = []
    for i in range(len(str)):
        if str[i] == start_c:
            for j in range(i+1, len(str)):
                num.append(str[j])
                if str[j] == end_c:
                    return ''.join(num[: -1])
    return None



##取前n个句子  
def get_first_n_sent(txtFile, n_sen):
    with open(in_path+'file.txt', 'r') as f:
        num = 0
        for line in f:
            ##处理句子
            
            num += 1
            if num == n_sen:
                break
    
    
##制作biGram句子
def make_biGram_wordList(wordsList):
    biGram_wordsList = wordsList[0: 2]
    if len(wordsList) >= 2:
        for i in range(2, len(wordsList)):
            biGram_word = wordsList[i-2] + wordsList[i-1]
            biGram_wordsList.append(biGram_word)
            biGram_wordsList.append(wordsList[i])
        biGram_wordsList.append(wordsList[-2] + wordsList[-1])

    return biGram_wordsList
    
    
    
#定位目标字符在字符串中的位置
str_ = '护手霜在微波炉里微热后涂抹更滋润'
char_='微波炉'
str_.find(char_)  #首次出现的位置

str_='小明买冰棍花了5元，买糖果花了3元，买游戏花了59元，小明今天一共花了67元。'
word = '花了'
a = [e.start() for e in re.finditer(word, str_)]  #全部位置


#文本的格式处理
content_str = re.sub('\s', ' ', content_str)   #一句话，非常简单


#去除数值，（将"数字"， ":"  替换成  ""）
wordCount = '雅诗兰黛:12'
re.sub("[0-9 \:]", "", wordCount)


#判断是否为中文
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

#str是否为数值
def is_num(str_):
    try:
        float(str_)
        return True
    except ValueError:
        return False
    
    
#提取括号内的字符
string = 'abe(ac)(ad)sjsjs(sss)'
p1 = re.compile(r'[(](.*?)[)]', re.S)  #最小匹配
re.findall(p1, string)


#多个字符的切分
text = '你好！吃早饭了吗？再见。'
re.split('。|！|？',text)



##编码和解码：
#常用的编码格式：'utf-8', 'gbk', 'unicode'
#编码：表示将字符串编码成'utf-8'的格式
str_ = '数说故事'
str_.encode(encoding='utf-8')  #输出：b'\xe6\x95\xb0\xe8\xaf\xb4\xe6\x95\x85\xe4\xba\x8b'


#解码：表示将经过'utf-8'编码字符串解码为正常格式。（）
line = b'\xe6\x95\xb0\xe8\xaf\xb4\xe6\x95\x85\xe4\xba\x8b'
line.decode(encoding='utf-8')  #encoding='utf-8'，需要解码的字符串也要是'utf-8'才行
                               #输出：'数说故事'
                               


##评价指标############################
from sklearn.metrics import (r2_score, mean_absolute_error, confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
#分类：
confusion_matrix(y_true, y_pred)  #混淆矩阵  
accuracy_score(y_true, y_pred)    #acc
precision_score(y_true, y_pred, average='macro')  #精确率
recall_score(y_true, y_pred, average='macro')  #召回率
f1_score(y_true, y_pred, average='macro')  #F1
def getAuc(y_true,y_pred):  #AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    aucs = auc(fpr,tpr)
    return aucs

#回归：
r2_score(y_true, y_pred)  #均方差损失
mean_absolute_error(y_true, y_pred)  #绝对值损失
    
    

##随机数##################################
import random
random.seed(10)  #确定随机种子，确保每次的结果都一样
np.random.seed(10)

#随机生成一个1~100的整数：
np.random.randint(1,100)
np.random.randint(0,100,(1000,128))   #生成的数据的shape是(1000, 128)

#随机打乱数据
#df
df = sklearn.utils.shuffle(df，random_state=2).reset_index(drop=True)
df = df.sample(frac=1，random_state=2).reset_index(drop=True)

#list
np.random.shuffle(list_)  #默认直接改变原list_，前面不能接 = 
random.shuffle(list_)



##python数据结构：字典，列表，集合，元祖###############################################################
###用字典记录元素出现的次数
#方法一：强行手写
elem_list = [1,1,1,3,2,2,2,3,4,5,5,4,4,3,3]
def cal_wordCount(elem_list):
    elemCount = {}
    for word in elem_list:
        if word not in elemCount.keys():
            elemCount[word] = 0  
        elemCount[word] += 1
        
    return elemCount
# 方法二：字典推导式 + .count()
elemCount = {e:elem_list.count(e) for e in elem_list}


##字典排序
sorted(elemCount.items(), key=lambda x: x[0], reverse=True)  #根据键
sorted(elemCount.items(), key=lambda x: x[1], reverse=True)  #根据值

elemCount_list = list(elemCount.items())
elemCount_list = sorted(elemCount_list, key=lambda x: x[1])  #列表也可以排序


##获取字典最大值对应的键
dict_ = {'a':1, 'b':0, 'c':9, 'd':4}

max(dict_, key=dict_.get)
max(dict_.keys(), key=lambda x: dict_[x])  #这个好一点


#获得“键”对应的值，如果键不在字典中，则返回'?'
idx2word.get(idx, '?')  


##找出一个序列中出现次数最多的元素
from collections import Counter
words = ['look', 'into', 'my', 'eyes', 'eyes', 'not', 'around', 'the',]
Counter(words).most_common(4)
# out: [('eyes', 2), ('look', 1), ('into', 1), ('my', 1)]


#集合###########################
new_set = set()  #新建集合

a_set = set([1,2,3,4,5])
b_set = set([4,5,6,7,8])

a_set | b_set  #并集：找所有的元素 
a_set & b_set  #交集：找共有的元素
a_set - b_set  #差集：找a独有的元素

a_set.add(e)     #往集合添加元素，与list的append类似
a_set.remove(e)  #移除某个元素


##非常非常重要的apply函数####################################：
##传单个df参数
#sentimentAnalysis(content)  #输入文字的内容，判断这段文字的情感正负
#sentimentAnalysis(content, title)  ##输入文字的内容和标题，判断情感正负

#注：sentimentAnalysis()是一个函数，参数为df_PG_29.content的每个元素
df_PG_29['hornbill'] = df_PG_29.content.apply(sentimentAnalysis)

##传多个df参数
df_PG_29['hornbill'] = df_PG_29.apply(lambda r: sentimentAnalysis(r['content'], r['title']), axis=1)

##传1个df参数和1个（或多个）其他参数


##传多个df参数和1个（或多个）其他参数


##异常#######################################################
#异常处理（出现异常，也照常执行）
try:
    print('')
except:
    pass

#在需要不断有网络传输的时候，常用这样的格式，保证传输的速度
max_tries=5
while max_tries>0:
    try:
        #编写需要执行的代码
        
        break
    except:
        max_tries -= 1
        

        
        
##W2V与D2V#####################################
def train_W2V(sentenList, file):
    w2vModel = gensim.models.Word2Vec(sentences=sentenList, iter=3, size=300, hs=0, negative=5, min_count=5, window=5)
                    ''' sentences=sentenList, #训练的语料库 
                        hs=0,  #0表示采用负例采样，1表示huffman
                        negative=5, #负例采样的个数
                        min_count=5,  #只取最少要出现5次的词
                        window=8,  #上下文窗口的大小
                        iter=epoch_num,  #训练的次数'''
    w2vModel.save(file)
    return w2vModel

    
def load_W2V(file, mode):
    if mode == 'rb':  #加载二进制文件
        w2vModel = Word2Vec.load(file)
    elif mode == 'rt':  #加载txt文件
        w2vModel = gensim.models.KeyedVectors.load_word2vec_format(file, binary=False)
    return w2vModel

    
##均值法构建“句子向量”：
def sentence_to_vector(sentence):
    words_list = sentence.split(' ')
    array = np.array([w2vModel[word] for word in words_list if word in w2vModel])
    df_SentenceVec = pd.Series(array.mean(axis=0))

    return df_SentenceVec

    

def train_D2V(d2vCorpus, embedSize=200, epoch_num=1):
    model_dm = Doc2Vec(d2vCorpus, min_count=1, window=3, size=embedSize, sample=1e-3, negative=5, workers=4)
    model_dm.train(d2vCorpus, total_examples=model_dm.corpus_count, epochs=epoch_num)
    model_dm.save("doc2vec.model")

    return model_dm
    
model_dm = Doc2Vec.load("doc2vec.model")


###全局和局部变量 | 引用和复制###########################################
#注意：python只有在“函数”，“类”里才会区分全局变量和局部变量，在if，for里面不会区分

##全局和局部变量
b = [1,2,3]

def func():   #copy全局变量b来使用，但不会改变外边的b
    a = b + [4,5,6]
    return a  #输出：a=[1,2,3,4,5,6], b=[1,2,3]
    
def func(b):   #只是参数传递，不会改变外面的b
    b = [0]
    a = b + [4,5,6]
    return a  #输出：a=[0,4,5,6], b=[1,2,3]
    
def func():   #里面的b是局部变量，与外面的b无关
    b = [0]
    a = b + [4,5,6]
    return a  #输出：a=[0,4,5,6], b=[1,2,3]
    

def func():   #声明b是全局变量，任意地方修改b，都会改变！
    global b  #注意，b不需要在函数外面先定义，“这在调试时非常方便！！！”
    b = [0]
    a = b + [4,5,6]
    return a  #输出：a=[0,4,5,6], b=[0]

func()



########################################################################################################################################

#枚举
for i, elem in enumerate(['a', 'b', 'c']):
    print(i, elem) #输出：0 a   1 b  2 c
	
	

#判断是否存在文件
os.path.exists('../expect_extraction_data/fenXiZiDuan_fenci.txt')   #返回True 或 False

##断言：断言如果为真，没事；否则异常，并显示需要的信息
assert len(X_train) == len(y_train), 'X_train和y_train的长度对不上'


#format
print('aa{}bb{}cc{}'.format(1, 'y', 3.3))   #输出为：'aa1bbycc3.3'



#查看变量占用的内存
import sys 
print(sys.getsizeof(rawText))  #以字节显示
print(sys.getsizeof(rawText) / 2**20)  #以M显示

a = [0] * 1
b = [0] * 2
print(sys.getsizeof(a))   #单位是字节Byte，结果72；因为初始化的时候附带其他的信息，所以比较大
print(sys.getsizeof(b))   #单位是字节Byte，结果80；说明一个list元素占8字节

'''
内存最小单位：bit

8bit = 1Byte
1024Byte = 1kb
1024kb = 1M

一个字符串占1个字节
list的一个元素占8字节
'''


#reshape函数
'''
np.reshape(arr, newshape)
    arr：需要改变形状的数组
    newshape：改变后的形状，可以是int或tuple；
              如果是int len，则结果是 1*len 的数组
              如果是tuple，则结果变为tuple的形状；
                  第一个维度可以是-1，这种情况下会根据其余维度和长度自动推断“该维度”。
'''
matrix = [[1,2], [4,5], [7,8]]
np.reshape(matrix, [2, 3])  #输出：[[1,2,4], [5,7,8]]
np.reshape(matrix, [-1, 2]) #输出：[[1, 2],[4, 5],[7, 8]]
np.reshape(matrix, [-1, ])  #输出：[1,2,3,4,5,6]


#zip并行迭代
moonList = [1, 2, 3]
dayList = [21, 22, 23]
for (moon, day) in zip(moonList, dayList):
    print(moon, day)





######
if __name__ == '__main__':
    print(1)
'''
if __name__ == '__main__'的意思是：当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行
''' 
    
##python查找的效率
##另外concat的速度 “奇慢无比” ；千万不要在循环中做大量的concat
if e in list_:    #O(n)的时间复杂度
if e in set_:     #O(1)的时间复杂度，这个是最优的，在大数据上查找时用set
if e in dict_:    #O(1)的时间复杂度


##计时：
start = time()
print(time() - start)


##json：很有用，可以极大地压缩空间
json.dumps(obj_data)  #将 对象 编译成 字符串
json.loads(str_data)  #还原回对象


##这句话总是想不起，哎呀！！
if __name__ == '__main__':
    print(1)


##特征重要度排序
feat_impo = sorted(zip(X_train.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True)


##K折交叉验证
def stratiKFoldCV(X_train1, y_train1):
    F1_list = []
    SK = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)
    #KF = KFold(n_splits=self.n_flod)  #如果不需要分层，就用这个
    for tra_idx, val_idx in SK.split(X_train1, y_train1):
        X_tra, X_val = X_train1[tra_idx], X_train1[val_idx]
        y_tra, y_val = y_train1[tra_idx], y_train1[val_idx]
        
        clf = svm.LinearSVC()
        clf.fit(X_tra, y_tra)

        y_pred = clf.predict(X_val)
        F1_list.append(f1_score(y_val, y_pred))
    
    return np.mean(F1_list)



    
    
##加载停用词
set(pd.read_csv(in_path+'chi_stop_words.txt', sep='\a', header=None, encoding='gbk')[0])  #直接用pandas加载很快！！

def get_stopWord(stop_path):
    with open(stop_path + 'chi_stop_words.txt', 'r') as f:
        for line in f:
            chi_stopWord.append(line.strip())
    return set(chi_stopWord)
            
    
    
##英文分句
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')    
splitSentence = sentence_tokenizer.tokenize('i love you! but i can')

def splitSentence(paragraph):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = tokenizer.tokenize(paragraph)
    return sents


    
###中文分词，去停用词
def split_words(text_line, stopwords):
    words = jieba.lcut(text_line, cut_all=False)
    words = [w for w in words if w not in stopwords]
    words = list(filter(lambda x:len(x)>1, words))  #去除单个词
    return words  #结果是“词列表”
    
###英文分词，去停用词
def split_words(text_line):
    lmtzr = nltk.WordNetLemmatizer()
    text_token = CountVectorizer().build_tokenizer()(text_line.lower())  #去掉标点，去单个词，小写化
    text_token = [w for w in text_token if w not in stopWords]  #去停用词，自己下一份词表吧！
    text_stem = [lmtzr.lemmatize(w) for w in text_token]  #词干归一化
    return text_stem

    

###序列编码，one-hot，归一化
def encode_count(df, encoder_list):
    lbl = LabelEncoder()
    for i in range(0, len(encoder_list)):
        str_column_name = encoder_list[i]
        df[[str_column_name]] = lbl.fit_transform(df[[str_column_name]])
    return df
    
def encode_onehot(df, oneHot_list):
    for i in range(0, len(oneHot_list)):
        str_column_name = oneHot_list[i]
        feature_df = pd.get_dummies(df[str_column_name], prefix=str_column_name)
        df = pd.concat([df.drop([str_column_name], axis=1), feature_df], axis=1)
    return df
    
def normalize(df, normalize_list):
    scaler = StandardScaler()
    for i in range(0, len(normalize_list)):
        str_column_name = normalize_list[i]
        df[[str_column_name]] = scaler.fit_transform(df[[str_column_name]])
    return df

    
###制作统计特征：其它统计方式还有[min, max, mean, sum, count, mode, nunique]
def add_feat_min(df_tar, df_fea, grou, stati, name, na=0):  #“位置” 参数必须放在 “关键字” 参数的前面
    add_tem = df_fea.groupby(grou)[stati].min().reset_index().rename(columns={stati[0]: name[0]})
    df_tar = pd.merge(df_tar, add_tem, on=grou, how='left').fillna(na)
    return df_tar

    
##特征选择
def RFECV_feature_sel(X_tra, y_tra, X_val, X_test):  
    clf = lgb.LGBMClassifier()  #clf可以随便换！！！
    selor = RFECV(clf, step=1, cv=3)
    selor = selor.fit(X_tra, y_tra)

    X_tra_sel = selor.transform(X_tra)
    X_val_sel = selor.transform(X_val)
    X_test_sel = selor.transform(X_test)
    
    return X_tra_sel, X_val_sel, X_test_sel
    
def Tree_feature_sel(X_tra, y_tra, X_val, y_val, X_test, sel_num): 
    clf = lgb.LGBMClassifier()
    clf.fit(X_tra, y_tra, eval_set=[(X_val, y_val)], 
            eval_metric='binary_logloss', early_stopping_rounds=100, verbose=500)

    feat_impo = sorted(zip(X_tra.columns, clf.feature_importances_), key=lambda x: x[1], reverse=True)
    sel_list = [feat[0] for feat in feat_impo[0: sel_num]]
    
    X_tra_sel = X_tra[sel_list]
    X_val_sel = X_val[sel_list]
    X_test_sel = X_test[sel_list]
        
    return feat_impo, X_tra_sel, X_val_sel, X_test_sel

    


    
def word_cluster(w2vModel, Cluter, wordSet):
    new_wordList = []
    vecList = []
    no_WordList = []
    for w in wordSet:
        if w in w2vModel:
            new_wordList.append(w)
            vecList.append(w2vModel[w])
        else: no_WordList.append(w)

    Cluter.fit(vecList)  #Cluter是聚类器，如kmean或DBSCAN
    labelList = Cluter.labels_

    word_cluster_dict = {}
    for i in range(0, len(labelList)):
        c = labelList[i]
        if c not in word_cluster_dict.keys():
            word_cluster_dict[c] = []
        word_cluster_dict[c].append(new_wordList[i])
        
    return word_cluster_dict, no_WordList
    
    
##linux操作##########################################################################################################    
'''
2、去到python/bin的文件夹，+空格+train.py，+空格+外部参数
    如：../../python/bin/python train.py --model_name ian --dataset dataStory --logdir ian_logs  #设置的外部参数

3、安装相应的包
先cd到python/bin的文件夹，然后pip install packgae
    如：[guangchuang@algox bin]$ ./pip3 install torch 
    或：[guangchuang@algox ABSA-PyTorch-master]$ ../../python/bin/pip install numpy  #看文件夹的位置便可
	
'''




##python高级用法
######################################################################################################################################################################
######################################################################################################################################################################
###函数的参数################################################################################################################################
##位置参数
def power(x, n):      #计算x的n次方
    s = 1
    for i in range(n):
        s *= x
    return s

    
##默认参数
def power(x, n=2):     ##计算x的任意次方，但默认是2次方
    s = 1
    for i in range(n):
        s *= x
    return s

    
##可变参数
#可变参数允许你传入0个或任意个参数，这些可变参数在函数调用时自动组装为一个“tuple”
def sum(*arr):   #计算元组的乘积
    print(arr)   #输出(1,2,3)，自动帮你把他用tumple包装起来了
    s = 1
    for e in arr:
        s *= e
    return s
    
sum(1,2,3)  #输出：6


##关键字参数
#而关键字参数允许你传入0个或任意个“含参数名”的参数，这些关键字参数在函数内部自动组装为一个“dict”
def preson_info(name, age, **kw):
    print(name, age, kw)

preson_info('liGuangChuang', 24)  #输出：liGuangChuang 24 {}
preson_info('liGuangChuang', 24, city='maoMing')  #输出：liGuangChuang 24 {'city': 'maoMing'}

##这4种参数的顺序：（位置、默认、可变、关键字）




###变量(参数)传递##################################################################################################################################
'''
变量的传递一开始都是“传递引用”，也即两个变量的内存空间是一样的。

如果参数是“可变对象”(如:list,dict,set)，且其中一个变量修改某个内部元素的值，则还是“传递引用”，
    也即这两个变量还是使用的是同一片内存，变量的值都是一样。
    
但直接给另一个变量赋一个新值（不管是可变对象，还是不可变对象），则变成“值传递”，
    也即开辟了新的内存空间，变量的改变互相不影响
    
如果变量传递一开始就使用.copy()，则指定是“值传递”
'''
#参数的传递，一开始都是引用：（也就是不会增加内存空间）
oringin = [1, 2, 3]  #123、'123'、[1,2,3]、(1,2,3)、{1:'a', 2:'b', 3:'c'}都一样
transmit = oringin
print(id(oringin))  #打印出的内存地址，一样
print(id(transmit))

##修改“可变对象”的某个内部元素，还是“传递引用”
oringin = [1, 2, 3]  
transmit = oringin
oringin[0] = 'what?'  
print(id(oringin))  #打印出的内存地址，还是一样
print(id(transmit))


##直接完全赋一个新的值，变成“传递值”
oringin = [1, 2, 3]       
transmit = oringin
oringin = [3,4,5]  #123、'123'、[1,2,3]、(1,2,3)、{1:'a', 2:'b', 3:'c'}都一样
print(id(oringin))  #打印出的内存地址，不同了
print(id(transmit))


oringin = '123'  
transmit = oringin
oringin = '456' ##不可变对象的修改只能是赋一个新值。。。
print(id(oringin))  #打印出的内存地址，不同了
print(id(transmit))


##使用.copy()，相当于指定一开始就是“值传递”
oringin = [1, 2, 3]  
transmit = oringin.copy()
oringin[0] = 'what?'  
print(id(oringin))  #打印出的内存地址，不同了
print(id(transmit))


##可变对象修改某个(某部分)元素，还是在自身上修改
oringin = [[1,2,3], [4,5,6], [7,8,9]]
for list_ in oringin:
    list_.append('what')

##输出：oringin == [[1,2,3,'what'], [4,5,6,'what'], [7,8,9,'what']]

oringin = [1,2,3]
oringin[3:] = [0,0,0,0,0]
##输出：oringin == [1,2,3,0,0,0,0,0]




###全局和局部变量##########################################################################################################################
#1、python只有在 “函数” 或 “类” 里才会区分全局变量和局部变量，在if，for里面不会区分
#2、在使用全局变量时，是否把这个全局变量作为参数传进来，无所谓
#3、在函数内部定义了变量（跟全局变量同名），那么这个变量跟外面的全局变量就无关了
#4、在函数内部，对变量进行索引操作（如b[0] = 999 ）、或用global声明这个变量为全局变量，会改变全局变量的值。

b = [1,2,3]

##外面的变量不会改变###########
def func0():  #或者func(b)
    a = b + [4,5,6]  #直接把外面的变量来用
	b = [0]
    return a   #报错，因为将b识别为局部变量，但这个局部变量又没有定义

def func1():  #或者func(b)
    a = b + [4,5,6]  #直接把外面的变量来用
    return a   

def func2():  #或者func(b)
    b = [0]  #创建了局部变量，外面的变量b跟这个无关
    a = b + [4,5,6]
    return a  #输出：a=[0,4,5,6], b=[1,2,3]


##要注意，会修改外部的变量#####
def func3():  #或者func(b)
    b[0] = 0  
    a = b + [4,5,6]
    return a  #输出：a=[0,2,3,4,5,6], b=[0,2,3]

def func4():   #声明b是全局变量，在函数内部修改b，外面的b也会改变！
    global b  #注意；在函数里面声明了外部变量，则可以直接修改
    b = [0]
    a = b + [4,5,6]
    return a  #输出：a=[000,4,5,6], b=[0]

func()
print(a, b)




###生成器#######################################################################################################################################
'''
其实呢很简单，没必要向网上说得那么专业。。。
就是程序执行到yeild时，就结果扔出去；其他都是正常执行，并不会像return那样终止程序的。

每次next(g),程序就执行到yield，然后下次再next(g)，程序从刚刚那yield的下一句开始执行。
用for i in g：相当于不断使用next(g)，直到程序运行结束。。。
'''
##函数生成器
def my_range(start, end, gap):
    i = start
    while i < end:
        yield i  
        i = i + gap

for i in my_range(0, 10, 1):
    print(i)
    
##类生成器
class my_range():
    def __init__(self, start, end, gap):
        self.start = start
        self.end = end
        self.gap = gap
        
    def __iter__(self):
        i = self.start
        while i < self.end:
            yield i
            i = i + self.gap
            
for i in my_range(0, 10, 1):
    print(i)
    
    
##生成器表达式：
g = (x*x for x in range(0, 10))
for e in g:
    print(e)

    
##深度学习中，bs的生成：
def batch_iter(X, y, bs):
    data_len = len(X)
    n_batch = int(data_len / bs)

    idx_shuffle = np.random.permutation(np.arange(data_len))  #随机打乱数据，而且X，y要对应
    X_shuffle = np.array([X[i] for i in idx_shuffle])
    y_shuffle = np.array([y[i] for i in idx_shuffle])

    for i_bat in range(n_batch):
        sta_idx = i_bat * bs
        end_idx = min((i_bat+1) * bs, data_len)
        yield X_shuffle[sta_idx: end_idx], y_shuffle[sta_idx: end_idx], i_bat
        
        
		
		
###装饰器#########################################################################################################################################
'''
装饰器用于，在不改变原来函数的基础上，扩展原来函数的功能
'''
##不使用装饰器
import time

def deco(func):
    start = time.time()
    func()
    run_time = (time.time() - start)
    print("time is {} ".format(run_time))

def func():
    for _ in range(50000000):
        pass
    print('end')

deco(func)


##使用最原始的装饰器
def deco(func):   #deco()就是一个装饰器，他的参数是函数，返回值也是函数
    def wrapper():
        start = time.time()
        func()
        run_time = (time.time() - start)
        print("time is {} ".format(run_time))
    return wrapper

@deco
def func():
    for _ in range(50000000):
        pass
    print('end')

func()   #直接执行func()函数便可


##使用“带指定参数”的装饰器
def deco(func):   
    def wrapper(a, b):
        start = time.time()
        func(a, b)
        run_time = (time.time() - start)
        print("time is {} ".format(run_time))
    return wrapper

@deco
def func(a, b):
    for _ in range(50000000):
        pass
    print(a+b, 'end')

func()   #直接执行func()函数便可


##(重点)，使用“可以接纳任意参数”的装饰器
def deco(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        run_time = (time.time() - start)
        print("time is {} ".format(run_time))
    return wrapper

@deco
def func(a, b, c, d, e):  #这里定义多少个参数都可以
    for _ in range(50000000):
        pass
    print(a+b+c+d+e, 'end')

func(1,2,3,4,5)   


##(重点)，使用多个装饰器
def deco_1(func):
    def wrapper(*args, **kwargs):
        print("this is deco_1")
        func(*args, **kwargs)
        print("deco_1 end here")

    return wrapper

def deco_2(func):
    def wrapper(*args, **kwargs):
        print("this is deco_2")
        func(*args, **kwargs)
        print("deco_2 end here")

    return wrapper

@deco_1
@deco_2
def func(a, b, c, d, e):  # 这里定义多少个参数都可以
    print("this is func")
    for _ in range(50000000):
        pass
    print(a + b + c + d + e, 'end')


func(1, 2, 3, 4, 5)

#输出：              执行顺序：deco_1 -> deco_2 -> func -> deco_2 -> deco_1
#this is deco_1 
#this is deco_2
#this is func
#15 end
#deco_2 end here
#deco_1 end here






