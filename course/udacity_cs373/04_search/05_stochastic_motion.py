# --------------
# USER INSTRUCTIONS
#
# Write a function called stochastic_value that 
# returns two grids. The first grid, value, should 
# contain the computed value of each cell as shown 
# in the video. The second grid, policy, should 
# contain the optimum policy for each cell.
#
# --------------
# GRADING NOTES
#
# We will be calling your stochastic_value function
# with several different grids and different values
# of success_prob, collision_cost, and cost_step.
# In order to be marked correct, your function must
# RETURN (it does not have to print) two grids,
# value and policy.
#
# When grading your value grid, we will compare the
# value of each cell with the true value according
# to this model. If your answer for each cell
# is sufficiently close to the correct answer
# (within 0.001), you will be marked as correct.

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>'] # Use these when creating your policy grid.

# ---------------------------------------------
#  Modify the function stochastic_value below
# ---------------------------------------------

def stochastic_value(grid,goal,cost_step,collision_cost,success_prob):
    failure_prob = (1.0 - success_prob)/2.0 # Probability(stepping left) = prob(stepping right) = failure_prob
    value = [[collision_cost for col in range(len(grid[0]))] for row in range(len(grid))]
    policy = [[' ' for col in range(len(grid[0]))] for row in range(len(grid))]
    
    r = len(grid)
    c = len(grid[0])

    value[goal[0]][goal[1]] = 0
    policy[goal[0]][goal[1]] = '*'

    changed = True
    while changed:
        changed = False
        for x in range(r):
            for y in range(c):
                if grid[x][y] == 1:
                    value[x][y] = collision_cost
                    continue
                    
                for idx, (d, d_name) in enumerate(zip(delta, delta_name)):
                    xn, yn = x + d[0], y + d[1]
                    if (xn >= 0) and (xn < r) and (yn >= 0) and (yn < c) and (grid[xn][yn] == 0):
                        cost_direct = cost_step + value[xn][yn]
                    else:
                        cost_direct = cost_step + collision_cost

                    d_left = delta[(idx - 1) % 4]
                    xn, yn = x + d_left[0], y + d_left[1]
                    if (xn >= 0) and (xn < r) and (yn >= 0) and (yn < c) and (grid[xn][yn] == 0):
                        cost_left = cost_step + value[xn][yn]
                    else:
                        cost_left = cost_step + collision_cost

                    d_right = delta[(idx + 1) % 4]
                    xn, yn = x + d_right[0], y + d_right[1]
                    if (xn >= 0) and (xn < r) and (yn >= 0) and (yn < c) and (grid[xn][yn] == 0):
                        cost_right = cost_step + value[xn][yn]
                    else:
                        cost_right = cost_step + collision_cost

                    cost = success_prob * cost_direct + failure_prob * cost_left + failure_prob * cost_right
                    if cost < value[x][y]:
                        value[x][y] = cost
                        changed = True
                        policy[x][y] = d_name

    return value, policy

# ---------------------------------------------
#  Use the code below to test your solution
# ---------------------------------------------

grid = [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 1, 0]]
goal = [0, len(grid[0])-1] # Goal is in top right corner
cost_step = 1
collision_cost = 1000
success_prob = 0.5

value,policy = stochastic_value(grid,goal,cost_step,collision_cost,success_prob)
for row in value:
    print(row)
for row in policy:
    print(row)

# Expected outputs:
#
#[471.9397246855924, 274.85364957758316, 161.5599867065471, 0],
#[334.05159958720344, 230.9574434590965, 183.69314862430264, 176.69517762501977], 
#[398.3517867450282, 277.5898270101976, 246.09263437756917, 335.3944132514738], 
#[700.1758933725141, 1000, 1000, 668.697206625737]


#
# ['>', 'v', 'v', '*']
# ['>', '>', '^', '<']
# ['>', '^', '^', '<']
# ['^', ' ', ' ', '^']
