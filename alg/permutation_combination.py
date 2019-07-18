# hello encoding=utf-8
"""https://www.quora.com/How-would-you-explain-an-algorithm-that-generates-permutations-using-lexicographic-ordering
https://github.com/CodingTrain/website/blob/master/CodingChallenges/CC_035.2_LexicographicOrder/Processing/CC_035_2_LexicographicOrder/CC_035_2_LexicographicOrder.pde
"""


def permutation(array):
    """ recursion 
    """
    if len(array) == 1:
        return [array]
    output = []
    for i in range(len(array)):
        if i == len(array) - 1:
            remain = array[0:i]
        else:
            remain = array[0:i] + array[i+1:]
        for back in permutation(remain):
            output.append([array[i]] + back)
    return output


def combine(array, n):
    """ recursion 
    """
    if len(array) == n:
        return [array]
    elif n == 0:
        return []
    output = []
    for i in range(len(array) - n + 1):
        remain = array[i+1:]
        if n - 1 == 0:
            output.append([array[i]])
        else:
            for r in combine(remain, n-1):
                output.append([array[i]] + r)
    return output


if __name__ == "__main__":
    array = [1, 2, 3, 4]
    for line in permutation(array):
        print(line)

    A = [2, 3, 5, 2]
    for line in combine(A, 2):
        print(line)
