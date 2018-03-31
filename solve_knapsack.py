# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 01:48:32 2018
Uses Python 3.6

@author: ariahklages-mundt
"""
import numpy as np
from itertools import chain, combinations

def solve_knapsack_DP(items, v, w, W):
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

def solve_knapsack_MIM(items, v, w, W):
    '''Solves 0-1 knapsack problem using meet-in-the-middle
    note: this can be faster than DP if W is large but n is not too large'''
    n = len(items)
    v_dict = {}
    w_dict = {}
    for i in range(n):
        v_dict[items[i]] = v[i]
        w_dict[items[i]] = w[i]
    A = items[:int(n/2.)]
    B = items[int(n/2.):]
    wvA = powerset_weights(A, v_dict, w_dict)
    wvB = powerset_weights(B, v_dict, w_dict)
    
    wvA, wvA_ind = simplify_subsets(wvA, W)
    wvB, wvB_ind = simplify_subsets(wvB, W)
    
    i,j, max_v = best_subsets(wvA, wvB, W)
    S1 = recover_subset(A, wvA_ind[i])
    S2 = recover_subset(B, wvB_ind[j])
    return max_v, list(S1+S2)

def solve_knapsack_approx(items, v, w, W, eps):
    '''approximate knapsack 0-1 solution by rescaling weights, eps in (0,1)
    In general, this is not FPTAS (there is another approx that achieves that), but it is if w=v'''
    n = len(items)
    K = eps*W/n
    wp = [int(np.ceil(w[i]/K)) for i in range(n)]
    Wp = int(np.floor(W/K))
    return solve_knapsack_DP(items,v,wp,Wp)

###############################################################################
    
def powerset(items):
    '''returns powerset iterator'''
    items = list(items)
    return chain.from_iterable(combinations(items,i) for i in range(len(items)+1))

def powerset_weights(A, v_dict, w_dict):
    '''returns weights and values of subsets of A'''
    psA = powerset(A)
    wvA = []
    for S in psA:
        wvA.append((sum([w_dict[s] for s in S]),sum([v_dict[s] for s in S])))
    dtype = [('weight', int),('value', int)]
    wvA = np.array(wvA, dtype=dtype)
    return wvA

def simplify_subsets(wvB, W):
    dtype = [('weight', int),('value', int)]
    wvB_r = np.array([(i['weight'],-i['value']) for i in wvB],dtype=dtype)
    
    #sort ascending by weight and, with tie, descending by value
    B_argsort = np.argsort(wvB_r,order=['weight','value'])
    del wvB_r
    wvBp = []
    wvBp_ind = []
    
    #for given weight, choose largest value subset + greater weight => greater subset value
    for i in B_argsort:
        if wvB[i]['weight'] <= W:
            if len(wvBp) == 0:
                wvBp.append(wvB[i])
                wvBp_ind.append(i)
            elif (wvB[i]['weight'] > wvBp[-1][0]) and (wvB[i]['value'] > wvBp[-1][1]):
                wvBp.append(wvB[i])
                wvBp_ind.append(i)
        else:
            break
    wvBp = np.array(wvBp, dtype=dtype)    
    return wvBp, wvBp_ind

def binary_search_B(w, wvB, W):
    '''search wvB for the highest weight/highest value entry that can be used
    wvB is sorted according to simplify_subsets, w is weight from subset of A'''
    first = 0
    last = len(wvB)-1
    best = 0 #the empty set (wvB[0]) works
    while first <= last:
        i = (first + last) // 2
        if wvB[i]['weight'] + w < W:
            first = i + 1
            best = i
        elif wvB[i]['weight'] + w > W:
            last = i - 1
        elif wvB[i]['weight'] + w == W:
            break
    if wvB[i]['weight'] + w > W:
        return best
    else:
        return i

def best_subsets(wvA, wvB, W):
    '''wvA and wvB are sorted according to simplify_subsets; returns indices of wvA, wvB'''
    best = (0, 0, 0) #the empty sets (wvA[0] and wvB[0]) work with weight 0
    for i in range(len(wvA)):
        j = binary_search_B(wvA[i]['weight'], wvB, W)
        val = wvA[i]['value'] + wvB[j]['value']
        if val > best[2]:
            best = (i, j, val)
    return best[0], best[1], best[2]

def recover_subset(A, i):
    '''return ith subset of powerset of A'''
    psA = powerset(A)
    for j in range(i+1):
        S = next(psA)
    return S

###############################################################################



#test Knapsack solvers
items = list(range(10))
W = 7897
v = [438, 33, 33857, 4729, 9, 19, 14, 310, 32, 44]
w = [438, 33, 33857, 4729, 9, 19, 14, 310, 32, 44]
max_v1, max_it1 = solve_knapsack_DP(items,v,w,W)
max_v2, max_it2 = solve_knapsack_MIM(items,v,w,W)
eps = 0.9
max_v3, max_it3 = solve_knapsack_approx(items,v,w,W,eps)
print(max_v1 == max_v2)
print(max_v3 > max_v2*eps)

items = list(range(10))
W = 5
w = [4,4,4,2,2,2,8,8,8,1]
v = [1,0,3,3,1,2,8,7,9,0]
max_v1, max_it1 = solve_knapsack_DP(items,v,w,W)
max_v2, max_it2 = solve_knapsack_MIM(items,v,w,W)
max_v3, max_it3 = solve_knapsack_approx(items,v,w,W,eps=0.9)
print(max_v1 == max_v2)
print(max_v3 > max_v2*eps)

#This example is hard for the exact solvers, but easy for the approximate solver
items = list(range(109))
W = 231417
w = [552, 16, 28642, 90, 2795, 761411, 73228, 2575, 9821, 8083, 40, 653, 1, 212, 27, 417, 3384, 1173, 148, 5, 253, 24, 11, 122, 356, 1, 1, 21, 2, 1343, 2, 67, 5950, 1, 550, 26, 3131, 4431, 5, 5034, 601, 2072, 27, 1, 16, 12, 4089, 22266, 915, 5, 2754, 90, 531, 8, 38, 1058, 1452, 407, 15021, 7, 72364, 518, 18483, 42, 9916, 205, 1651, 82, 1, 9370, 2598, 743, 8164, 555, 4177, 50, 20, 1110, 2182, 7, 223, 1090, 1, 85, 12, 654, 12, 1334, 444, 10, 102, 579, 1171, 240, 1767, 1433, 28, 50, 1190, 793, 167, 537, 1321, 1938, 1051, 27833, 9, 4634, 10081, 62, 26]
v = w[:]
max_v3, max_it3 = solve_knapsack_approx(items,v,w,W,eps=0.3)
