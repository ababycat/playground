while True:
    try:
        input_str = input().strip()
        out = []

        split_ok = False
        start = 0
        stop = 0
        while not split_ok:

            start = stop
            length = 8

            stop = start + length
            if stop > len(input_str):
                stop = len(input_str)
                split_ok = True

            result = input_str[start:stop]
            result = result + '0'*(8-(stop-start))
            # print(result)
            out.append(result)

        for i in out:
            print(i)
        
    except:
        break