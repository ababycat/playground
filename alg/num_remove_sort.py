n = int(input())
num_list = []
for i in range(n):
    num = int(input())
    num_list.append(num)

t = list(set(num_list))
t.sort()

for p in t:
    print(p)