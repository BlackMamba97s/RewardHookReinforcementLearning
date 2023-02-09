import random

c = 0
for i in range(0, 9):
    x = random.randint(0, 9)
    if x == 0:
        c+=1

print(c/i)
