# NBA 108:85 共有多少种顺序

r = 108
c = 85

dp = [[0 for i in range(0, c + 1)] for j in range(r + 1)]

dp[0][0] = 0

dp[0][1] = 1
dp[0][2] = 2
dp[0][3] = 4

dp[1][0] = 1
dp[2][0] = 2
dp[3][0] = 4

for i in range(0, r + 1):
    for j in range(0, c + 1):
        if (i == 0) and (j <= 3):
            continue
        if (j == 0) and (i <= 3):
            continue
        if j - 1 >= 0:
            dp[i][j] += dp[i][j-1]
        if j - 2 >= 0:
            dp[i][j] += dp[i][j-2]
        if j - 3 >= 0:
            dp[i][j] += dp[i][j-3]
        if i - 1 >= 0:
            dp[i][j] += dp[i-1][j]
        if i - 2 >= 0:
            dp[i][j] += dp[i-2][j]
        if i - 3 >= 0:
            dp[i][j] += dp[i-3][j]

print(dp[i][j])



