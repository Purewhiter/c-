from time import*
num = 100
i = 2
qh = 0
start = perf_counter()
for i in range(2, 100):
    j = 2
    for j in range(2, i):
        if(i % j == 0):
            break
    else:
        qh += i
print(qh)
delta = perf_counter()-start
print(delta)
