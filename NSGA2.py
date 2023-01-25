# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 13:31:19 2022

@author: dell
"""
#Importing required modules
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tools import nsga结果计算函数


#First function to optimize 换acc
def function1(X_test, Y_test, Y_prep, protected_attribute_name,protected_value):
    acc,aod = nsga结果计算函数(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)
    value = 1-acc
    return value

#Second function to optimize 换aod
def function2(X_test, Y_test, Y_prep, protected_attribute_name,protected_value):
    acc,aod = nsga结果计算函数(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)
    value = abs(aod)
    return value

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf
    return sorted_list

#Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front

#Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

# Function to carry out the crossover
def crossover(a,b):
    r=random.random()
    if r>0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)

#Function to carry out the mutation operator
def mutation(solution,Y_prep):
    for i in range(len(solution)):
        for j in range(len(solution[i])):            
            mutation_prob = random.random()
            if mutation_prob <1:
                solution[i][j] = random.randint(0,len(Y_prep))

    return solution

# def mutation(Y_prep,solution):
#     Y_prepm = []
    
#     for i in range(len(solution)):
#         Y_prep2 = np.array(Y_prep)
#         # Y_prep2 = list(Y_prep2[:,])
#         for j in solution[i]:
#             Y_prep2[j] =  1-Y_prep2[j]
        
#         Y_prepm.append(Y_prep2)
    
#     return Y_prepm

# def generatesol(X_test,Y_prep, protected_attribute_name):
#     X_test[protected_attribute_name]
    
    # return solution
def nsga(X_test, Y_test, Y_prep, protected_attribute_name,protected_value):
    
    #Main program starts here
    pop_size = 20
    max_gen = 100
    
    #Initialization
    # min_x=-55
    # max_x=55
    
    
    solution=[[random.randint(0,len(Y_prep)-1) for i in range(0,pop_size)] for i in range(0,pop_size)]
    
    # solution = []
    
    gen_no=0
    
    while(gen_no<max_gen):
        
        
        
        function1_values = [function1(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)for i in range(0,pop_size)]
        function2_values = [function2(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)for i in range(0,pop_size)]
    
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    
        # print("The best front for Generation number ",gen_no, " is")
        # for valuez in non_dominated_sorted_solution[0]:
        #     print(round(solution[valuez],3),end=" ")
        # print("\n")
    
        # crowding_distance_values=[]
        # for i in range(0,len(non_dominated_sorted_solution)):
        #     crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
        solution2 = solution[:]
        
        #Generating offsprings
    
        while(len(solution2)!=2*pop_size):
        #     a1 = random.randint(0,pop_size-1)
        #     b1 = random.randint(0,pop_size-1)
            for i in range(len(solution)):    
                solution2.append(mutation(solution[i],Y_prep))
                
        function1_values2 = [function1(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)for i in range(0,pop_size)]
        function2_values2 = [function2(X_test, Y_test, Y_prep, protected_attribute_name,protected_value)for i in range(0,pop_size)]
    
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])
        crowding_distance_values2=[]
    
        for i in range(0,len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
    
        new_solution= []
        for i in range(0,len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if(len(new_solution)==pop_size):
                    break
            if (len(new_solution) == pop_size):
                break
    
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1

    return Y_prep
