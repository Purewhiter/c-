def up(rate):
    base = 1
    for i in range(365):
        if i % 7 in [0, 1]:
            base = base*0.99
        else:
            base = base*(1+rate)
    return base


rate = 0.01
while up(rate) < 37.78:
    rate += 0.001
print("工作日的努力参数是:{:.3f}".format(rate))
