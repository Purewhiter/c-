count = 0


def hanoi(n, start, mid, end):
    global count
    if n == 1:
        print(str(1), ':', start, "to", end)
        count += 1
    else:
        hanoi(n-1, start, end, mid)  # 执行 h(2, 'A', 'C', 'B')
        print(str(n), ':', start, "to", end)
        count += 1
        hanoi(n-1, mid, start, end)  # 执行 h(2, 'B', 'A', 'C')
    return count


s = hanoi(3, "A", "B", "C")
print(s)
