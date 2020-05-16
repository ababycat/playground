# ----------
# User Instructions:
# 
# Implement the function optimum_policy2D below.
#
# You are given a car in grid with initial state
# init. Your task is to compute and return the car's 
# optimal path to the position specified in goal; 
# the costs for each motion are as defined in cost.
#
# There are four motion directions: up, left, down, and right.
# Increasing the index in this array corresponds to making a
# a left turn, and decreasing the index corresponds to making a 
# right turn.

forward = [[-1,  0], # go up
           [ 0, -1], # go left
           [ 1,  0], # go down
           [ 0,  1]] # go right
forward_name = ['up', 'left', 'down', 'right']

# action has 3 values: right turn, no turn, left turn
action = [-1, 0, 1]
action_name = ['R', '#', 'L']

# EXAMPLE INPUTS:
# grid format:
#     0 = navigable space
#     1 = unnavigable space 
grid = [[1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1, 1]]


init = [4, 3, 0] # given in the form [row,col,direction]
                 # direction = 0: up
                 #             1: left
                 #             2: down
                 #             3: right
                
goal = [2, 0] # given in the form [row,col]
# goal = [2, 5] # given in the form [row,col]

cost = [2, 1, 20] # cost has 3 values, corresponding to making 
                  # a right turn, no turn, and a left turn
# cost = [20, 1, 2]

# EXAMPLE OUTPUT:
# calling optimum_policy2D with the given parameters should return 
# [[' ', ' ', ' ', 'R', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', '#'],
#  ['*', '#', '#', '#', '#', 'R'],
#  [' ', ' ', ' ', '#', ' ', ' '],
#  [' ', ' ', ' ', '#', ' ', ' ']]
# ----------

# ----------------------------------------
# modify code below
# ----------------------------------------

def check_edge(grid, x, y):
    if (x >= 0) and (x < len(grid)) and (y >= 0) and (y < len(grid[0])) and (grid[x][y] == 0):
        return False
    return True

def move_forward(state):
    x, y, d = state
    nx = x + forward[d][0]
    ny = y + forward[d][1]
    return nx, ny, d

def move_step(state, a, grid):
    a = action[a]
    if a == action[1]:
        nx, ny, nd = move_forward(state)
    else:
        x, y, d = state
        # nd = [0, 1, 2, 3][(d+a) % 4]
        nd = (d + a) % 4
        nx, ny, nd = move_forward([x, y, nd])
                
    if check_edge(grid, nx, ny):
        return False, None
    else:
        return True, [nx, ny, nd]
        
def optimum_policy2D(grid,init,goal,cost):
    r, c = len(grid), len(grid[0])
    
    value = [[[999 for x in range(4)] for x in range(c)] for y in range(r)]
    policy = [[[' ' for x in range(4)] for x in range(c)] for y in range(r)]
    policy2D = [[' ' for x in range(c)] for y in range(r)]

    # x, y = goal
    # policy2D[x][y] = '*'

    # x, y, d = init
    # for d in range(4):
    #     value[x][y][d] = 0
    
    # changed = True
    # while changed:
    #     changed = False
    #     for x in range(r):
    #         for y in range(c):
    #             for d in range(4):
    #                 if (x == init[0]) and (y == init[1]):
    #                     if value[x][y][d] > 0:
    #                         changed = True
    #                         value[x][y][d] = 0
    #                         print('-------')

    #                 for a in range(3):
    #                     state = [x, y, d]
    #                     reachable, new_state = move_step(state, a, grid)
    #                     if reachable:
    #                         nx, ny, nd = new_state
    #                         if cost[a] + value[x][y][d] < value[nx][ny][nd]:
    #                             value[nx][ny][nd] = cost[a] + value[x][y][d]
    #                             policy[x][y][d] = action_name[a]
    #                             changed = True

    x, y = goal
    policy2D[x][y] = '*'
    for i in range(4):
        value[x][y][i] = 0
    
    changed = True
    while changed:
        changed = False
        for x in range(r):
            for y in range(c):
                for d in range(4):
                    if value[goal[0]][goal[1]][d] > 0:
                        value[goal[0]][goal[1]][d] = 0
                        changed = True

                    for a in range(3):
                        state = [x, y, d]
                        reachable, new_state = move_step(state, a, grid)
                        if reachable:
                            nx, ny, nd = new_state
                            if cost[a] + value[nx][ny][nd] < value[x][y][d]:
                                value[x][y][d] = cost[a] + value[nx][ny][nd]
                                policy[x][y][d] = action_name[a]
                                changed = True

    print(value[init[0]][init[1]])
    print(value[goal[0]][goal[1]])

    x, y, d = init
    while policy2D[x][y] != '*':
        policy2D[x][y] = policy[x][y][d]
        unreachable, new_state = move_step([x, y, d], action_name.index(policy2D[x][y]), grid)
        x, y, d = new_state
    return policy2D

for line in optimum_policy2D(grid,init,goal,cost):
    print(line)