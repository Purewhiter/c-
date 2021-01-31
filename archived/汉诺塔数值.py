def hn(n):
    if n == 1:
        return 1
    else:
        return hn(n-1)*2+1


print(hn(9))
