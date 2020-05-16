# 对于一个数字序列，请设计一个复杂度为O(nlogn)的算法，返回该序列的最长上升子序列的长度，
# 这里的子序列定义为这样一个序列U1，U2...，其中Ui < Ui+1，且A[Ui] < A[Ui+1]。

# 给定一个数字序列A及序列的长度n，请返回最长上升子序列的长度。
# 测试样例：

# [2,1,4,3,1,5,6],7

# 返回：4


# -*- coding:utf-8 -*-

class AscentSequence:
    # def findLongest(self, A, n):
    #     # write code here
    #     lis = [[] for x in range(n)]
    #     lis[0].append(A[0])

    #     for i in range(1, n):
    #         for j in range(0, i):
    #             if (A[i] > lis[j][-1]):
    #                 lis[i] = lis[j].copy()
    #                 # break
    #         lis[i].append(A[i])

    #     max_len = 0
    #     for word in lis:
    #         print(word)
    #         max_len = len(word) if len(word) > max_len else max_len
    #     return max_len

    def findLongest(self, A, n):
        # write code here
        LIS = [1 for x in range(n)]
        INDEX = [-1 for x in range(n)]
        for i in range(1, n):
            for j in range(0, i):
                if A[i] > A[j]:
                    LIS[i] = max(LIS[i], LIS[j]+1)
                    INDEX[i] = j
        max_val = max(LIS)
        max_index = LIS.index(max_val)
        idx = max_index
        res = []
        while idx != -1:
            res.append(A[idx])
            idx = INDEX[idx]
        return max_val, res[::-1]


    def findLongest2(self, A, n):
        # write code here 
        # O(N*log(N))
        from bisect import bisect_left

        if n == 1:
            return 1
        # 存放长度为i的候选序列的最大值
        B = [float("inf") for x in range(n)]
        # 存放长度为i的候选序列，最后一个值使用最小值
        C = [[] for x in range(n)]
        B[0] = A[0]
        C[0] = [A[0]]
        L = 1
        for i in range(1, n):
            if A[i] < B[0]:
                B[0] = A[i]
                C[0] = [A[i]]
            elif A[i] > B[L-1]:
                B[L] = A[i]
                C[L] = C[L-1].copy()
                C[L].append(A[i])
                L = L + 1
                if L >= n:
                    break
            elif (A[i] > B[0]) & (A[i] < B[L-1]):
                idx = bisect_left(B, A[i])
                B[idx] = A[i]
                C[idx][-1] = A[i]
            else:
                continue
        return L, C[L-1]

        # for i in range(0, n):
        #     if A[i] < B[0]:
        #         B[0] = A[i]
        #         C[0].append(A[i])
        #     else:
        #         j = bisect_left(B, A[i])
                


if __name__ == "__main__":
    solution = AscentSequence()
    A = [1, 2, 3, 4, 5]
    # max_len, index = solution.findLongest(A, len(A))
    max_len, index = solution.findLongest2(A, len(A))
    print("max len %d" % max_len)
    print("index: ", index)