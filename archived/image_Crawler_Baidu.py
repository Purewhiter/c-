# -*- coding: utf-8 -*-
# @Time    : 2020/3/3 0029 00:30
# @Author  : NanXun
# @File    : image_Crawler_baidu.py
# @Software: PyCharm

import requests
import re
import os

header = {
    "Cookie": "BDqhfp=%E8%83%AD%E8%84%82%E9%B1%BC%26%26-10-1undefined%26%260%26%261; BAIDUID=92EEF1EC8355E0EE9DE5E63A51A13439:FG=1; BIDUPSID=92EEF1EC8355E0EE9DE5E63A51A13439; PSTM=1581736398; H_PS_PSSID=1438_21120_30824_26350; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; delPer=0; PSINO=6; BDRCVFR[dG2JNJb_ajR]=mk3SLVN4HKm; userFrom=www.baidu.com; BDRCVFR[-pGxjrCMryR]=mk3SLVN4HKm; indexPageSugList=%5B%22%E8%83%AD%E8%84%82%E9%B1%BC%22%2C%22%E5%BE%AE%E4%BF%A1%E5%B0%8F%E7%A8%8B%E5%BA%8F%20%E7%82%B9%E5%87%BB%E6%8C%89%E9%92%AE%E5%90%8E%E5%AD%97%E5%8F%91%E7%94%9F%E5%8F%98%E5%8C%96%22%2C%22%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%AB%96%E5%9B%BE%22%2C%22%E4%BA%BA%E8%84%B8%E8%AF%86%E5%88%AB%22%2C%22%E6%80%A7%E5%88%AB%E8%AF%86%E5%88%AB%22%2C%22%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%AB%96%E5%9B%BE%22%2C%22%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%22%2C%22%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%22%2C%22%E7%AE%80%E5%8D%95%E8%83%8C%E6%99%AF%E7%B4%A0%E6%9D%90app%22%5D; cleanHistoryStatus=0",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36"
}


def getHTMLText(url, pn):
    try:
        r = requests.get(url, headers=header, timeout=30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        print("获取第{0}页内容成功！".format(pn+1))
        # print(r.text)
        # r,content.decode() utf-8编码
        return r.text
    except:
        print("获取第{0}页内容失败！".format(pn+1))


def getImageList(html, count):
    ilist = re.findall(r'"objURL":"(.*?)"', html)
    if ilist:
        print("共获取" + str(len(ilist)) + "张图片：")
        for i in range(len(ilist)):
            try:
                print("正在解析第{0}张图片链接：".format(count+1), end="")
                if(ilist[i][-3:] == 'jpg'or'png'or'PNG'or'JPG' or ilist[i][-4:] == 'jpeg'):
                    print(ilist[i])
                    imgList.append(ilist[i])
                    count = count + 1
            except:
                print('解析失败，该图片链接不存在')
    return imgList


def saveImageList(imgList, root):
    for index, img in enumerate(imgList):
        filepath = root + "/" + str(index) + ".jpg"
        # print(root)
        #filepath = root + "/" + str(i) + "." + imgList[i][-3:]
        # print(filepath)
        # print(imgList[i])
        try:
            if not os.path.exists(root):
                os.mkdir(root)  # 创建目录
            if not os.path.exists(filepath):
                r = requests.get(img)
                with open(filepath, 'wb') as f:  # wb 只读二进制文件
                    # print(r.content)
                    f.write(r.content)  # HTTP 响应内容的二进制形式
                    f.close()
                    print("已下载第{0}张图片".format(index+1))
            else:
                print("图片已存在")
        except:
            print("图片下载失败")


if __name__ == '__main__':
    count = 0
    imgList = []
    keyword = input("输入你想下载的图片的关键词：")
    print("****一次爬取将获得60张图片****")
    num = int(input("输入爬取图片的次数："))
    root = "C:\\Users\\Purew\\Desktop\\爬虫图片".format(keyword)  # 图片存储的地址
    for pn in range(num):
        try:
            pageurl = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word={0}&pn={1}'.format(
                keyword, 20 * pn)
            html = getHTMLText(pageurl, pn)
            # print(html)
            imgList = getImageList(html, count)
            # print(imgList[0])
            print("第{0}页爬取成功！".format(pn+1))
        except:
            print("第{0}页爬取失败！".format(pn+1))
    try:
        saveImageList(imgList, root)
        print("文件已保存至路径！")
        print("本次爬取结束！")
    except:
        print("文件保存失败请重试！")
