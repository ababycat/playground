"""You're going to write a binary search function.
You should use an iterative approach - meaning
using loops.
Your function should take two inputs:
a Python list to search through, and the value
you're searching for.
Assume the list only has distinct elements,
meaning there are no repeated values, and 
elements are in a strictly increasing order.
Return the index of value, or -1 if the value
doesn't exist in the list."""

def binary_search(input_array, value):
    """Your code goes here."""
    if value < input_array[0]:
        return -1, 0
    if value > input_array[-1]:
        return -1, n
    left, right = 0, len(input_array) - 1
    mid = (left + right) / 2
    while left <= right:
        if value == input_array[mid]:
            return 1, mid
        elif value > input_array[mid]:
            left = mid + 1
        elif value < input_array[mid]:
            right = mid - 1
        mid = (left + right) / 2
    return -1, left

test_list = [1,3, 5,9,11]
test_val1 = 0
test_val2 = 6
print binary_search(test_list, test_val1)
print binary_search(test_list, test_val2)