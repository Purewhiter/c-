'''
Author      : PureWhite
Date        : 2020-05-30 23:26:03
LastEditors : PureWhite
LastEditTime: 2020-12-31 19:43:40
Description : 
'''
s="你好吗好吗"
p=""
for i in s:
    p+=i
    if len(p)==4:
        print(p)
        p=p[1:]