#Modify the move function to accommodate the added 
#probabilities of overshooting or undershooting 
#the intended destination.

p=[0, 1, 0, 0, 0]
world=['green', 'red', 'red', 'green', 'green']
measurements = ['red', 'green']
pHit = 0.6
pMiss = 0.2
pExact = 0.8
pOvershoot = 0.1
pUndershoot = 0.1

def sense(p, Z):
    q=[]
    for i in range(len(p)):
        hit = (Z == world[i])
        q.append(p[i] * (hit * pHit + (1-hit) * pMiss))
    s = sum(q)
    for i in range(len(q)):
        q[i] = q[i] / s
    return q

def move(p, U):
    q = []
    q = [0 for x in range(len(p))]
    for i in range(len(p)):
        if p[i] > 0:
            q[(i+U)%len(p)] += pExact * p[i]
            q[(i+U-1)%len(p)] += pUndershoot * p[i]
            q[(i+U+1)%len(p)] += pOvershoot * p[i]
    return q
    
for i in range(100):
    p = move(p, 1)
    print(p)