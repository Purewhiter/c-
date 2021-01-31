
import jieba
with open(r"C:\Users\Purew\Desktop\沉默的羔羊.txt", "r", encoding="utf-8") as f:
    txt = f.read()
txtCut = jieba.lcut(txt)
txtDict = {}
for word in txtCut:
    if len(word) > 2:
        txtDict[word] = txtDict.get(word, 0)+1
items = list(txtDict.items())
items.sort(key=lambda x: x[1], reverse=True)
word, count = items[0]
print(word)
