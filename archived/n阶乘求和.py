def sum(n):
    m = 1
    s = 0
    for i in range(1, n+1):
        m *= i
        s += m
    return s


n = int(input())
print(sum(n))
