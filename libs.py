import re
import numpy as np
import sys
import getopt
import json
import math
#import Queue
import networkx as nx
import pylab
import numpy as np
import logging, sys


def init(filename):
    """
    This function read the tgff file and
    build computation matrix, communication matrix, rate matrix.
    TGFF is a useful tool to generate directed acyclic graph, tfgg file represent a task graph.
    """
    f = open(filename, 'r')

    #Get hyperperiod
    hyperperiod=int(f.readline().split()[1])
    #print(hyperperiod)
    
    f.readline()
    f.readline()
    f.readline()
    f.readline()

    # Calculate the amount of tasks
    num_of_tasks = 0
    while f.readline().startswith('\tTASK'):
        num_of_tasks += 1
    #print('Number of tasks =',num_of_tasks)

    # Build a communication matrix
    data = [[-1 for i in range(num_of_tasks)] for i in range(num_of_tasks)]
    line = f.readline()
    while line.startswith('\tARC'):
        line = re.sub(r'\bt\d_', '', line)
        i, j, d = [int(s) for s in line.split() if s.isdigit()]
        data[i][j] = d
        line = f.readline()
    #for line in data:
    #    print(line)

    while not f.readline().startswith('# type'):
        pass


    # Build a computation matrix
    comp_cost = {}
    line = f.readline()
    while line.startswith('\t'):
        comp_cost.update({int(line.split()[0]):int(line.split()[1])})
        line = f.readline()
    #for key in comp_cost.keys():
    #    print(key,comp_cost[key],type(key),type(comp_cost[key]))

    

    return [hyperperiod, num_of_tasks, data, comp_cost]


def Get_Neighborhood(position,radius,M,N): #return a list which consists of positions around input position with radius=r
    row=int(position/N)
    col=position%N
    neighborhood=[]
    for i in range(row-radius,row+radius+1):
        if(i>=0 and i<M):
            for j in range(col-radius,col+radius+1):
                if(j>=0 and (i!=row or j!=col) and j<N):
                    neighborhood.append(i*N+j)
    return neighborhood

def Get_mapping_reward(PEs_task_current_solution,computation_ability,M,N):
    ret=0
    for i in range(0,len(PEs_task_current_solution)):
        if(len(PEs_task_current_solution[i])): #this PE has tasks
            ret+=computation_ability[int(i/N)][i%N]
    return ret

def Get_detailed_data(num_of_tasks,edges,comp_cost):#输出邻接矩阵,total_needSend,total_needReceive,execution
    adj_matrix=np.zeros((num_of_tasks+1,num_of_tasks+1),dtype=np.int)#task从1开始算
    for i in range(0,len(edges)):
        for j in range(0,len(edges[i])):
            if(edges[i][j]!=-1):
                adj_matrix[i+1][j+1]=comp_cost[edges[i][j]]#adj_matrix[i][j]不为0，表示task i有到task j的出边，数组的值为待传输的量

    total_needSend=np.zeros(num_of_tasks+1,dtype=np.int)#task从1开始算的
    total_needReceive=np.zeros(num_of_tasks+1,dtype=np.int)#task从1开始算的
    for i in range(1,num_of_tasks+1):
        task_i_needSend=0
        for j in range(1,num_of_tasks+1):
            task_i_needSend+=adj_matrix[i][j]
        total_needSend[i]=task_i_needSend

    for j in range(1,num_of_tasks+1):
        task_j_needReceive=0
        for i in range(1,num_of_tasks+1):
            task_j_needReceive+=adj_matrix[i][j]
        total_needReceive[j]=task_j_needReceive

    execution=np.zeros(num_of_tasks+1,dtype=np.int)#task从1开始算的
    for i in range(1,num_of_tasks+1):
        execution[i]=comp_cost[i-1]
    
    return adj_matrix,total_needSend,total_needReceive,execution

def find_start_task(adj_matrix,num_of_tasks):#寻找入度为0的点
    ret=[]
    in_degree=np.zeros(num_of_tasks+1,dtype=np.int)
    for i in range(1,num_of_tasks+1):
        for j in range(1,num_of_tasks+1):
            if(adj_matrix[i][j]!=0):
                in_degree[j]+=1
    for i in range(1,num_of_tasks+1):
        if(in_degree[i]==0):
            ret.append(i)
    return ret

def get_sorted_dict(dict):
    ret={}
    #l=sorted(dict.keys())
    l=[]
    for i in dict.keys():
        l.append(int(i))
    l.sort()
    for i in l:
        ret.update({str(i):dict[str(i)]})
    return ret
    

if __name__ == '__main__':
    hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N12_autocor.tgff')
    adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
    print(adj_matrix)
    print(total_needSend)
    print(total_needReceive)
    print(execution)
    print(find_start_task(adj_matrix,num_of_tasks))
