#!/usr/bin/env python3
# Viterbi Ave minimization problem
# based on skeleton code by D. Crandall, March 2019

import numpy as np

# problem definition
home_elevations = [10, 12, 15, 1, 25, 0, 1, 16, 16, 18]
N = 10 # cost of raising/lowering a home 1 foot
M = 1  # constact factor in cost of staircases

possible_elevations = range(0, 26)
home_count = len(home_elevations)

# left_messages[i] is the message *to* the i-th home, from the i+1-th home
# right_messages[i] is the message to the i-th home, from the i-1-th home
left_messages = np.zeros((home_count, 26),)
right_messages = np.zeros((home_count, 26),)


def D(my_elevation, i):
    return N*abs(home_elevations[i]-my_elevation) ## done

def V(my_elevation, neighbor_elevation):
    return M*(my_elevation-neighbor_elevation)*(my_elevation-neighbor_elevation) ## done

for iteration in range(0, home_count*2):
    new_left_messages = np.zeros((home_count, 26),)
    new_right_messages = np.zeros((home_count, 26),)

    for i in range(0, home_count-1):
        for neighbor_elevation in possible_elevations:
            new_right_messages[i+1][neighbor_elevation] = np.min([(D(x,i)+V(x,neighbor_elevation)+right_messages[i][x]) for x in possible_elevations])  ## done

    for i in range(1, home_count):
        for neighbor_elevation in possible_elevations:
            new_left_messages[i-1][neighbor_elevation] =  np.min([(D(x,i)+V(x,neighbor_elevation)+left_messages[i][x]) for x in possible_elevations])  ## done

    np.copyto(left_messages, new_left_messages)
    np.copyto(right_messages, new_right_messages)

# finally, every home chooses its best elevation based on the last set of neighbors and its own D() cost
new_elevations = [0] * home_count
for i in range(0, len(home_elevations)):
    new_elevations[i] = np.argmin([(right_messages[i][my_elevation] + left_messages[i][my_elevation] + D(my_elevation, i)) for my_elevation in possible_elevations ] )

# calculate the cost of the final answer
cost = (new_elevations[0] - home_elevations[0])*N
for i in range(1, len(home_elevations)):
    cost += abs(new_elevations[i] - home_elevations[i])*N + M*(new_elevations[i] - new_elevations[i-1])**2

print("Problem inputs:")
print("   Cost of raising/lowering yard: $%d/foot" % N)
print("   Cost of staircase            : height squared x $%d" % M)
print("   Current yard elevations      : " + str(home_elevations))
print("")
print("Solution:")
print("   Min cost to install sidewalk : $%d" % cost)
print("   New yard elevations          : " + str(new_elevations))



