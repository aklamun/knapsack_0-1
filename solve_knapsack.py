# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 01:48:32 2018
Uses Python 3.6

@author: ariahklages-mundt
"""
import numpy as np

def solve_knapsack(items, v, w, W):
    '''Solves 0-1 knapsack problem using dynamic programming, returns the maximum objective value and the corresponding set of items
    items is list of items, v and w are lists of corresponding values and weights, W is weight capacity; v,w, and W should have int entries
    note: this can be quite slow if n,W are large'''
    n = len(items)
    assert isinstance(W, int)
    for i in range(n):
        assert isinstance(v[i], int)
        assert isinstance(w[i], int)
    
    m = np.zeros((n+1, W+1),dtype=int)
    ml = {}
    for j in range(W+1):
        m[0,j] = 0
        ml[(0,j)] = []
    
    for i in range(1,n+1):
        for j in range(W+1):
            if w[i-1] > j:
                m[i,j] = m[i-1, j]
                ml[(i,j)] = ml[(i-1,j)]
            else:
                if m[i-1,j] > m[i-1,j-w[i-1]] + v[i-1]:
                    m[i,j] = m[i-1,j]
                    ml[(i,j)] = ml[(i-1,j)]
                else:
                    m[i,j] = m[i-1,j-w[i-1]] + v[i-1]
                    ml[(i,j)] = ml[(i-1,j-w[i-1])] + [i-1]
    
    return m[n,W], [items[i] for i in ml[(n,W)]]



#test Knapsack solver
items = range(10)
W = 7897
v = [438, 33, 33857, 4729, 9, 19, 14, 310, 32, 44]
w = [438, 33, 33857, 4729, 9, 19, 14, 310, 32, 44]
max_v, max_it = solve_knapsack(items,v,w,W)