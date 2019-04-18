#!/usr/bin/env python3
import networkx as nx
import matplotlib.pyplot as plt
import copy
import numpy as np

G = nx.grid_2d_graph(5, 5)

# create edges between east-west neighbors
for i in range(0, 5):
    for j in range(0, 4):
        if(i==j):G[(i,j)][(i,j+1)]['weight'] = 0
        else:G[(i,j)][(i,j+1)]['weight'] = 200# SET THIS WEIGHT

# create edges between north-south neighbors
for i in range(0, 4):
    for j in range(0, 5):
        if(i==j):G[(i,j)][(i+1,j)]['weight'] = 0
        else:G[(i,j)][(i+1,j)]['weight'] = 500 # SET THIS WEIGHT
    
# create edges between source and sink
for i in range(0, 5):
    for j in range(0, 5):
        G.add_edge('s', (i,j), weight=0)
        G.add_edge('t', (i,j), weight=0)

# You'll probably want to change some of the edge weights between
# source and sink, e.g.:
G['t'][(1,1)]['weight']=10000
G['t'][(3,2)]['weight']=10
G['t'][(4,0)]['weight']=10000
G['s'][(2,2)]['weight']=10000
G['s'][(3,0)]['weight']=500

# Run min cuts. 
(cost, (set1, set2)) = nx.minimum_cut(G, 's', 't', capacity='weight')

print("Cost of labeling: $ %d" % cost)
if('s' not in set1):(set2, set1) = (set1, set2)

for i in range(0, 5):print("".join(['D' if (i,j) in set1 else 'R' for j in range(0,5)]))

