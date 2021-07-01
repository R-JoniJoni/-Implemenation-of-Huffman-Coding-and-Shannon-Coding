'Keyriri'
import numpy as np
import math
import sys
from decimal import Decimal

from numpy.lib.shape_base import expand_dims

def dTob(n, pre=4): #把一个带小数的十进制数n转换成二进制小数点后面保留pre位小数
    string_number1 = str(n) #number1 表示十进制数，number2表示二进制数
    flag = False   
    for i in string_number1: #判断是否含小数部分
        if i == '.':
            flag = True
            break
    if flag:   #在本实验中，string_number1的小数点左边一定是0
        decimal = Decimal(str(n))
        decimal_convert = ""
        
        i = 0  
        #while decimal != 0 and i < pre:  
        while i < pre:
            result = int(decimal * 2)  
            decimal = decimal * 2 - result
            decimal_convert = decimal_convert + str(result)
            i = i + 1  
        string_number2 = '0' + '.' + decimal_convert

        return string_number2
    else:   #若十进制只有整数部分, 在本实验中只能为0
        string_number = '0'

        while(pre > 1):
            string_number += '0'
            pre -= 1

        return '0.' + string_number

def cal_average_code_length(r, P):  #计算平均码长
    L = 0
    for n in range(1, r + 1):
        L += float(P[1, n]) * len(P[2, n])
    
    return L

def add_dummy_symbols(Q, r, P):  #添加dummy symbols
    N = Q
    if(Q <= 1):
        print('You should input an integer larger than 1')
        sys.exit()
        
    while(N < r):
        N += (Q - 1)
    delt = N - r
    for n in range(0, delt):
        r += 1
        P = np.insert(P, r, 0, axis=1)  #对矩阵进行列拓展，并在新拓展的列填入0
        P[0, r] = 'dummy' + str(n)      #对所有的dummy symbols的命名都以dummy打头
        P[2, r] = ''

    return r, P

def delete_dummy_symbols(P, r): #删除dummy symbols
    length = r
    for n in range(1, length + 1):
        temp = P[0, (length + 1 - n)]
        if(temp[0 : 5] == 'dummy'): #之前对dummy symbol的命名都是以dummy打头的，这里以此判断该symbol是否为dummy symbol
            r -= 1
            P = np.delete(P, (length + 1 - n), axis=1)
    return r, P

def cal_entropy_and_P(text):  #返回文本的entropy以及一个包括了symbol, possibility, codeword的矩阵
    states = np.array([])               #状态合集
    transition_matrix = np.array([])    #状态概率转移矩阵，其横纵坐标与列表states中的顺序相同
    num_states = 0  #总状态数
    num = len(text) #文本的长度
    entropy = 0

    #先把第一个状态读进来
    states = np.append(states, text[0])
    transition_matrix = np.array([1.])
    num_states = 1

    #接下来读入剩下的状态
    for a in text[1:]:
        if(a in states):    #目前读入的状态是之前记录过的
            [[index]] = np.argwhere(states == a)
            transition_matrix[index] = int(transition_matrix[index]) + 1 #状态概率转移矩阵先不存“概率”，而是先存“出现的次数”，
                                                                         #等遍历完所有状态后，会把“出现的次数”除以相应的各行的总数，从而得到“概率”
        else:   #目前读入的状态不是之前记录过的
            states = np.append(states, a)
            transition_matrix = np.append(transition_matrix, 1.)
            num_states = num_states + 1
    for n in range(0, num_states):
        transition_matrix[n] = transition_matrix[n] / num #得到“概率”
    P = np.array([states, transition_matrix])   #此时P变成了2 * num_states的数组

    #为避免概率太小时，进行计算会出现误差，若一个symbol概率太小，则将其改为0.0001
    for n in range(0, num_states):
        if(float(P[1, n]) > 0.0001):
            #P[1, n] = P[1, n][0 : 10]
            P[1, n] = P[1, n][0 : 10]
        else:
            P[1, n] = '0.00010000'
    P = np.insert(P, 2, '', axis = 0)
    P = np.insert(P, 0, 0, axis = 1)
    P[0, 0] = 'SYMBOLS'
    P[1, 0] = 'POSSIBILITY'
    P[2, 0] = 'CODEWORD'

    for n in range(1, num_states + 1):
        entropy -= math.log(float(P[1, n]), 2) * float(P[1, n])
        
    return entropy, P

def sort_matrix(r, P): #改变原来的数组，将其降序概率排序
    count = 0
    P_ = P.copy()
    #P_ = P
    for n in range(0, r):
        count = 0
        for m in range(2, r - n + 1):
            if(P_[1, m - 1] < P_[1, m]):
                temp = P_[2, m - 1]
                P_[2, m - 1] = P_[2, m]
                P_[2, m] = temp
                temp = P_[1, m - 1]
                P_[1, m - 1] = P_[1, m]
                P_[1, m] = temp
                temp = P_[0, m - 1]
                P_[0, m - 1] = P_[0, m]
                P_[0, m] = temp
                
                count += 1
        if(not count):
            break
    return P_

def copy_code(P, P_, r):    #把P_中的codeword填到P中SYMBOL相同的位置
    for n in range(1, r + 1):
        for m in range(1, r + 1):
            if(P[0, n] == P_[0, m]):
                P[2, n] = P_[2, m]

def cal_Huffman_codewords(Q, r, P):  #计算Huffman码。 Q-ary, r symbols of soure, P is the probability distribution
    if(r <= Q): #如果r<=Q，可以直接编码
        code = []
        for n in range(0, r):
            code.append(str(n))
            P[2, n + 1] = str(n)
    else:   #如果r>Q，把概率最小的Q个symbol合为一个，再把这个小的数组递归调用
        P_ = sort_matrix(r, P)  #概率降序排序

        #把概率最小的Q个symbol合为一起
        P1 = np.delete(P_, (r - Q + 2, r), axis=1)
        P1[0, r - Q + 1] = r
        for n in range(r - Q + 2, r + 1):
            P1[1, r - Q + 1] = float(P_[1, n]) + float(P1[1, r - Q + 1])
        
        cal_Huffman_codewords(Q, r - Q + 1, P1)
        for n in range(1, r - Q + 1):
            P_[2, n] = P1[2, n]
        for n in range(r - Q + 1, r + 1):
            P_[2, n] = P1[2, r - Q + 1] + str(n - r + Q - 1)
        
        copy_code(P, P_, r)

def cal_Shannon_codewords(r, P): #计算Shannon码。 r symbols of source, P is the probability distribution
    code_length = []#存储码长
    F= []           #F存储相应的概率和
    P_ = sort_matrix(r, P)
    
    F.append(0)
    for n in range(2, r + 1):
        F.append(0)
        for m in range(1, n):
            F[n - 1] = F[n - 1] + float(P_[1, m])
    #计算码长
    for n in range(0, r):
        code_length.append(math.ceil(-1 * math.log2(float(P_[1, n + 1]))))

    #填充Shannon码到矩阵中
    for n in range(0, r):
        P_[2, n + 1] = dTob(F[n], code_length[n])[2:]

    copy_code(P, P_, r)

def func_print_matrix(name, r, P):    #计算平均码长，并打打印出平均码长、symbol对应的编码
    L = cal_average_code_length(r, P)
    print('----------------------------------------------------')

    print(name)
    print('平均码长 =', L)
    
    print('符号和编码')
    for n in range(1, r + 1):
        print('[', end = '')
        print(P[0, n] + '|' + P[1, n] + '|' + P[2, n], end = '')
        print(']')

if __name__ == '__main__':
    #读入文本
    f = open('Steve_Jobs_Speech.txt', 'r', encoding = 'utf-8')
    #f = open('test.txt', 'r', encoding = 'utf-8')
    text = f.read()
    text = text.replace('\n', '')  #把文本中的换行消除
    f.close()

    

    #计算熵、矩阵、symbol数量
    entropy, possibility_distribution = cal_entropy_and_P(text)
    row, r = possibility_distribution.shape
    r -= 1

    #计算Huffman码
    print('Please input the Q-ary of Huffman code, Q = ', end = '')
    Q = input()
    Q = int(Q)
    print('ENTROPY =', entropy)
    r, possibility_distribution = add_dummy_symbols(Q, r, possibility_distribution) #填充dummy symbols
    cal_Huffman_codewords(Q, r, possibility_distribution)
    r, possibility_distribution = delete_dummy_symbols(possibility_distribution, r) #删除dummy symbols
    func_print_matrix('Huffman', r, possibility_distribution)   #计算平均码长，输出相应信息

    #计算Shannon码
    cal_Shannon_codewords(r, possibility_distribution)
    func_print_matrix('Shannon', r, possibility_distribution)   #计算平均码长，输出相应信息

    
    