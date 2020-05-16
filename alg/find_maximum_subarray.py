# hello encoding=utf-8

import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def time_wrapper(func):
    def wrapper(A):
        start = datetime.now()
        max_val, low, high = func(A)
        end = datetime.now()
        cost_time = (end-start).total_seconds()
        return max_val, low, high, cost_time
    return wrapper


@time_wrapper
def find_maximum_subarray_brute(A):
    max_val = A[0]
    max_index_i = 0
    max_index_j = 0
    for i in range(len(A)):
        sum_val = 0
        for j in range(i, len(A)):
            sum_val = sum_val + A[j]
            if sum_val > max_val:
                max_val = sum_val
                max_index_i = i
                max_index_j = j
    return max_val, max_index_i, max_index_j


def find_maximum_crossing_subarray(A, low, mid, high):
    max_left = mid
    left_sum = -sys.float_info.max
    sum_val = 0
    for i in range(mid, low-1, -1):
        sum_val += A[i]
        if sum_val > left_sum:
            left_sum = sum_val
            max_left = i
    
    max_right = mid
    right_sum = -sys.float_info.max
    sum_val = 0
    for i in range(mid+1, high+1):
        sum_val += A[i]
        if sum_val > right_sum:
            right_sum = sum_val
            max_right = i
    return max_left, max_right, left_sum + right_sum


def find_maximum_subarray_in(A, low, high):
    if low == high:
        return low, high, A[low]
    mid = int((low + high) / 2)
    left_low, left_high, left_sum = find_maximum_subarray_in(A, low, mid)
    right_low, right_high, right_sum = find_maximum_subarray_in(A, mid+1, high)
    left_mid, right_mid, mid_sum = find_maximum_crossing_subarray(A, low, mid, high)
    if (left_sum >= right_sum) & (left_sum >= mid_sum):
        return left_low, left_high, left_sum
    elif (right_sum >= left_sum) & (right_sum >= mid_sum):
        return right_low, right_high, right_sum
    else:
        return left_mid, right_mid, mid_sum


@time_wrapper
def find_maximum_subarray_dc(A):
    left, right, val = find_maximum_subarray_in(A, 0, len(A)-1)
    return val, left, right


@time_wrapper
def find_maximum_subarray_linear(A):
    """
    for 循环中的第一个判断决定是否一个在j处的最大子数组包括A[j]
    The first test within the for loop determines 
    whether a maximum subarray ending at index j contains just A[j]. 
    当我门进入循环中的一次迭代, ending-here-sum存储j-1处最大子数组的和
    As we enter an iteration of the loop, 
    ending-here-sum has the sum of the values in a maximum 
    subarray ending at j-1. 
    如果ending-here-sum+A[j]>A[j],那么我们扩展最大子数组j-1去包含j
    If ending-here-sum+A[j]>A[j], 
    then we extend the maximum subarray ending at index j+1
    to include index j. 
    (if中的判断仅仅是两边都减掉A[j])
    (The test in the if statement just subtracts out A[j] from both sides.) 
    否则,我们开始一个新的子数组,它的低和高都是j,和是A[j]
    Otherwise, we start a new subarray at index j, 
    so both its low and high ends have the value j and 
    its sum is A[j]. 
    我们知道了在j处的最大子数组后,
    我们判断这个数组是否超过之前找到的最大子数组,不论该子数组以i或小于j的结尾
    Once we know the maximum subarray ending at index j, 
    we test to see whether it has a greater sum than the maximum 
    subarray found so far, 
    ending at any position less than or equal to j. 
    如果成立,那么我们更新最低和最高和max-sum
    If it does, then we update low, high, 
    and max-sum appropriately.
    
    Since each iteration of the for loop takes constant time, 
    and the loop makes n iterations, 
    the running time of MAX-SUBARRAY-LINEAR is O(n).

    MAX-SUBARRAY-LINEAR(A)
        n = A.length
        max-sum = -inf
        ending-here-sum = -inf
        for j = 1 to n
            ending-here-high = j
            if ending-here-sum > 0
                ending-here-sum = ending-here-sum + A[j]
            else ending-here-low = j
                ending-here-sum = A[j]
            if ending-here-sum > max-sum
                max-sum = ending-here-sum
                low = ending-here-low
                high = ending-here-high
        return (low, high, max-sum)"""
    n = len(A)
    max_sum = -float("inf")
    # ending_here_sum = -sys.float_info.max
    ending_here_sum = -float("inf")
    for j in range(0, n):
        ending_here_high = j
        if ending_here_sum > 0:
            ending_here_sum = ending_here_sum + A[j]
        else:
            ending_here_low = j
            ending_here_sum = A[j]
        if ending_here_sum > max_sum:
            max_sum = ending_here_sum
            low = ending_here_low
            high = ending_here_high
    return max_sum, low, high


def test1():
    A = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]

    print('method: ', '-'*50, find_maximum_subarray_brute.__name__)
    max_val, max_index_i, max_index_j = find_maximum_subarray_brute(A)
    max_array = A[max_index_i:max_index_j] + [A[max_index_j]]
    print("max val is: %d, start i: %d, end j: %d" % (max_val, max_index_i, max_index_j))
    print("max array: ", max_array)

    print('method: ', '-'*50, find_maximum_subarray_dc.__name__)
    max_val, max_index_i, max_index_j = find_maximum_subarray_dc(A)
    max_array = A[max_index_i:max_index_j] + [A[max_index_j]]
    print("max val is: %d, start i: %d, end j: %d" % (max_val, max_index_i, max_index_j))
    print("max array: ", max_array)

    print('method: ', '-'*50, find_maximum_subarray_linear.__name__)
    max_val, max_index_i, max_index_j = find_maximum_subarray_linear(A)
    max_array = A[max_index_i:max_index_j] + [A[max_index_j]]
    print("max val is: %d, start i: %d, end j: %d" % (max_val, max_index_i, max_index_j))
    print("max array: ", max_array)


def test2():
    time_list = []
    n_list = [x for x in range(100, 5000, 100)]
    for length in n_list:
        np.random.seed(17)
        choice = np.random.permutation(length)
        nums = np.arange(-int(length/2), -int(length/2)+length)
        A = list(nums[choice])

        max_val1, max_index_i1, max_index_j1, cost_time1 = find_maximum_subarray_brute(A)
        max_array1 = A[max_index_i1:max_index_j1] + [A[max_index_j1]]

        max_val2, max_index_i2, max_index_j2, cost_time2 = find_maximum_subarray_dc(A)
        max_array2 = A[max_index_i2:max_index_j2] + [A[max_index_j2]]

        max_val3, max_index_i3, max_index_j3, cost_time3 = find_maximum_subarray_linear(A)
        max_array3 = A[max_index_i3:max_index_j3] + [A[max_index_j3]]

        if (max_index_i1 != max_index_i2) or (max_index_i1 != max_index_i3) or \
            (max_index_i2 != max_index_i3) or (max_index_j1 != max_index_j3) or \
                (max_index_j1 != max_index_j3) or (max_index_j2 != max_index_j3):
            print("different equal: " % length)

        time_list.append([cost_time1, cost_time2, cost_time3])
    
    time_list = np.array(time_list)
    plt.plot(n_list, time_list[:, 0], label=['brute'])
    plt.plot(n_list, time_list[:, 1], label=['dc'])
    plt.plot(n_list, time_list[:, 2], label=['linear'])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # test1()
    test2()