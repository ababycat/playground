n = input()
while n != 0:
    input_val = n
    result = 0

    while True:
        this_val = input_val // 3
        sum_val = this_val + input_val % 3
        input_val = sum_val
        result = result + this_val

        if sum_val <= 3:
            stop =  True
            if sum_val == 3:
                adder = 1
            elif sum_val == 2:
                adder = 1
            elif sum_val == 1:
                adder = 0
            result = result + adder
        else:
            stop = False

        if stop:
            print(result)
            break
    try:
        n = input()
    except:
        n = 0