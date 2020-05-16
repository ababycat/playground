def fb(n):
    if n <= 1:
        res = n
        return res
    res = fb(n-1) + fb(n-2)
    return res

print(fb(1))

def fb2(n):
    last, l_last = 1, 1
    print(1)
    print(1)
    for i in range(2, n):
        res = last + l_last
        l_last = last
        last = res
        print(res)
    return res
print('--')
print(fb2(3))


def factorial(n):
    res = 1
    for i in range(1, n+1):
        res = res*i
    return res

print(factorial(4))