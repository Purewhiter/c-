import time
scale = 20
print("执行开始".center(scale//2, "-"))
start = time.perf_counter()
for i in range(scale+1):
    a = (i/scale)*100
    b = '█'*i
    c = ' '*(scale-i)
    dur = time.perf_counter()-start
    print("\r{:.2f}%[{}->{}]{:.2f}s".format(a, b, c, dur), end="")
    time.sleep(0.05)
print("\n"+"执行结束".center(scale//2, "-"))
