'''
Author      : PureWhite
Date        : 2020-05-30 23:26:03
LastEditors : PureWhite
LastEditTime: 2020-08-28 13:25:19
Description : 
'''

def 输出斐波那契数列(n):
    if n == 1 or n == 2:
        return 1
    else:
        a = b = 1
        for i in range(1, n+1):
            c = a+b
            a = b
            b = c
            print(c, end=' ')
