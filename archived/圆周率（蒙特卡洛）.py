from random import random
from time import perf_counter
DARTS = 100*100


def count():
    hits = 0.0
    for i in range(1, DARTS+1):
        x, y = random(), random()
        dist = pow(x**2+y**2, 0.5)
        if dist <= 1.0:
            hits += 1
    pi = (hits/DARTS)*4
    return pi


t = 0.0
for i in range(100):
    t += count()
t /= 100
print(t)
