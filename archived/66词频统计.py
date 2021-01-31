#取文档
def gettxt():
    with open(r'‪C:\Users\Purew\Desktop\hamlet.txt','r') as txt:
        txt=txt.lower()
    for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
        txt=txt.replace(ch," ")
    return txt

#分词
wordlist=txt.split()

#统计词频
worddic={}
for word in wordlist:
    worddic[word]=worddic.get(word,0)+1

#排序

