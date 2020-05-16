"""Implement quick sort in Python.
Input a list.
Output a sorted list."""
def quicksort(array):
    def partition(A, p, r):
        i = p - 1
        for j in range(p, r):
            if A[j] <= A[r]:
                i = i + 1
                A[i], A[j] = A[j], A[i]
        A[i+1], A[r] = A[r], A[i+1]
        return i + 1

    def quicksort_in(array, p, r):
        if p < r:
            q = partition(array, p, r)
            quicksort_in(array, p, q-1)
            quicksort_in(array, q+1, r)

    quicksort_in(array, 0, len(array)-1)
    return array

test = [21, 4, 1, 3, 9, 20, 25, 6, 21, 14]
# test = [1, 3, 4, 5, 6, 7, 8, 9]
print quicksort(test)
