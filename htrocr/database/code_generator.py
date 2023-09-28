import random

numbers = ""

for i in range(1000):
    no = str(random.randint(0, 1000000))
    numbers += f"No. {no}\n"

with open('numbers.txt', 'w') as w:
    w.write(numbers)