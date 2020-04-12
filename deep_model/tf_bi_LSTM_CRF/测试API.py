from API import seq_out
from time import time


with open('D:/python_code/fenXiZiDuan.txt', 'rt') as rf:
    with open('D:/python_code/zhenshi.txt', 'wt', encoding='utf-8') as wf:
        for i, line in enumerate(rf):
            wf.write(line)
            if i == 200000000000:
                break

##############################################################################
seq_out = seq_out(mode='SW')



###################################################################
text_lines = ['罗东萍，我真的很喜欢你的哦，你在十七中读书吗？',
              '上次回了一趟母校，第三中学的变化还真是大啊！',
              '你在哪里读书，华南理工大学吗？',
              '真的死的很惨啊，你在广州待不下去了',
              '天气不错啊，想去哪里玩吗？',
              '哪里都不想去，就想在家待着',
              '我记得上次的事情还没搞清楚，你能不能搞清楚了再说',]
result1 = seq_out.out(text_lines)
print(result1)

###############################################################
##真实的测试环境啊！！！！！这个有点叼了！！！！

start = time()
result2 = seq_out.out('D:/python_code/zhenshi.txt', 'D:/python_code/SW2.txt')
print(result2)
print(time() - start)






