import random

months = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
dates = ""
# Style 0
for i in range(1000):
    year = str(random.randint(1800, 1950))
    month = random.choice(months)
    day = str(random.randint(1,31))
    dates += f"{day}/{month} {year}\n"
# Style 1
for i in range(1000):
    year = str(random.randint(1800, 1950))
    month = random.choice(months)
    day = str(random.randint(1,31))
    dates += f"{day} {month} {year}\n"
# Style 2
for i in range(1000):
    year = str(random.randint(1800, 1950))
    month = random.choice(months)
    day = str(random.randint(1,31))
    dates += f"{day}.{month}.{year}\n"
# Style 3
months = ['januar', 'februar', 'marts', 'april', 'maj', 'juni', 'juli', 'august', 'september', 'oktober', 'november', 'december']
for i in range(1000):
    year = str(random.randint(1800, 1950))
    month = random.choice(months)
    day = str(random.randint(1,31))
    dates += f"{day} {month} {year}\n"
# Style 4
months = ['jan.','febr.','marts','april','maj','juni','juli','aug.','sept.','okt.','nov.','dec.']
for i in range(1000):
    year = str(random.randint(1800, 1950))
    month = random.choice(months)
    day = str(random.randint(1,31))
    dates += f"{day} {month} {year}\n"
# Style 5
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
for i in range(1000):
    year = str(random.randint(1800, 1950))
    month = random.choice(months)
    day = str(random.randint(1,31))
    dates += f"{day} {month} {year}\n"
# Style 6
for i in range(1000):
    year = str(random.randint(1800, 1950))
    month = random.choice(months)
    day = str(random.randint(1,31))
    dates += f"{day}/{month} {year}\n"
# Style 7
for i in range(1000):
    year = str(random.randint(1800, 1950))
    month = random.choice(months)
    day = str(random.randint(1,31))
    dates += f"{day}-{month}-{year}\n"

with open('dates.txt', 'w') as w:
    w.write(dates)