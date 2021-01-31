def prime(n):
    flag = 1
    for i in range(2, int(n/2)+1):
        if n % i == 0:
            flag = 0
            break
    if flag == 1:
        print("yes")
    else:

        print("no")


prime(4)
