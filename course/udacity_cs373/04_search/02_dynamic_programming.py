# ----------
# User Instructions:
# 
# Create a function compute_value which returns
# a grid of values. The value of a cell is the minimum
# number of moves required to get from the cell to the goal. 
#
# If a cell is a wall or it is impossible to reach the goal from a cell,
# assign that cell a value of 99.
# ----------

grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0]]
goal = [len(grid)-1, len(grid[0])-1]
cost = 1 # the cost associated with moving from a cell to an adjacent one

delta = [[-1, 0 ], # go up
         [ 0, -1], # go left
         [ 1, 0 ], # go down
         [ 0, 1 ]] # go right

delta_name = ['^', '<', 'v', '>']

def compute_value(grid,goal,cost):
    # ----------------------------------------
    # insert code below
    # ----------------------------------------
    def near_val(value, x, y):
        r, c = len(value), len(value[0])
        output = []
        for d in delta:
            xn, yn = x + d[0], y + d[1]
            if (xn >= 0) & (xn <= len(grid)-1) & (yn >= 0) & (yn <= len(grid[0])-1):
                output.append(value[xn][yn])
        return output
        
    closed = [[0 for x in range(len(grid[0]))] for x in range(len(grid))]
    value = [[99 for x in range(len(grid[0]))] for x in range(len(grid))]
    
    x, y = goal
    value[x][y] = 0
    closed[x][y] = 1
    pos_list = []
    pos_list.append([x, y])
    t = 0
    while len(pos_list) > 0:
        for i in range(len(pos_list)):
            x, y = pos_list.pop(0)
            if not ((x == goal[0]) & (y == goal[1])):
                value[x][y] = min(near_val(value, x, y)) + 1
            closed[x][y] = 1
            for d in delta:
                xn, yn = x + d[0], y + d[1]
                if (xn >= 0) & (xn <= len(grid)-1) & (yn >= 0) & (yn <= len(grid[0])-1):
                    if grid[xn][yn] == 1:
                        value[xn][yn] = 99
                    elif closed[xn][yn] == 0:
                        t = t + 1
                        closed[xn][yn] = 1
                        pos_list.append([xn, yn])

    print(t)
    # make sure your function returns a grid of values as 
    # demonstrated in the previous video.
    return value 
    
for line in compute_value(grid,goal,cost):
    print(line)
