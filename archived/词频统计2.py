'''
Author      : PureWhite
Date        : 2020-05-30 23:26:03
LastEditors : PureWhite
LastEditTime: 2020-09-10 12:05:10
Description : 
'''
def getText():
    txt = open(r"C:\Users\Purew\Desktop\hamlet.txt", "r").read()
    txt = txt.lower()
    for char in '!"#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
        txt = txt.replace(char, ' ')
    return txt


hamtxt = getText()
hamlist = hamtxt.split()
hamdic = {}
for word in hamlist:
    hamdic[word] = hamdic.get(word, 0)+1
items = list(hamdic.items())
items.sort(key=lambda x: x[1], reverse=True)
for i in range(10):
    word, count = items[i]
    print(word)
