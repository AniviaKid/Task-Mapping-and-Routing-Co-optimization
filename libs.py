import re
import numpy as np
import sys
import getopt
import json
import math
import networkx as nx
import pylab
import logging, sys

import random
import copy


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from OnlineCompute1 import onlineTimeline


def init(filename):
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

def get_sorted_dict(dict):#将task_graph按照task的序号排序，再传入online_compute
    ret={}
    l=[]
    for i in dict.keys():
        l.append(int(i))
    l.sort()
    for i in l:
        ret.update({str(i):dict[str(i)]})
    return ret
    


def Get_full_route_by_XY(part_route,source_position,dest_position,num_of_rows):
    ret=copy.deepcopy(part_route)
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows
    cur_row=-1
    cur_col=-1

    if(len(part_route)==0):
        cur_row=int(source_position/num_of_rows)
        cur_col=source_position%num_of_rows
    else:#计算出数据现在走到了哪个位置
        cur_position=part_route[-1][0]
        cur_row=int(cur_position/num_of_rows)
        cur_col=cur_position%num_of_rows
        if(part_route[-1][1]=='N'):
            cur_row-=1
        elif(part_route[-1][1]=='S'):
            cur_row+=1
        elif(part_route[-1][1]=='W'):
            cur_col-=1
        elif(part_route[-1][1]=='E'):
            cur_col+=1

    while(cur_col<dest_col):
        tmp=[]
        tmp.append(cur_row*num_of_rows+cur_col)
        tmp.append('E')
        ret.append(tmp)
        cur_col+=1
    while(cur_col>dest_col):
        tmp=[]
        tmp.append(cur_row*num_of_rows+cur_col)
        tmp.append('W')
        ret.append(tmp)
        cur_col-=1

    while(cur_row<dest_row):
        tmp=[]
        tmp.append(cur_row*num_of_rows+cur_col)
        tmp.append('S')
        ret.append(tmp)
        cur_row+=1
    while(cur_row>dest_row):
        tmp=[]
        tmp.append(cur_row*num_of_rows+cur_col)
        tmp.append('N')
        ret.append(tmp)
        cur_row-=1
    
    
    return ret
    
def Get_reward_by_pendTimes(pendTimes):
    #pendTimes<=7时才能得到非负的reward
    return 7-pendTimes


#state为[state_tensor,cur_position,partRouteFromRL]，传进来的partRoute的格式是直接的路由表，没有第一位第二位的task
def check_if_Done(state,source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult):#检查当前的state是否已经结束，结束了的话直接把end_state,reward,done=True返回
    next_state_tensor=state[0]
    next_position=-1
    next_partRoute=state[2]

    cur_row=int(state[1]/num_of_rows)
    cur_col=state[1]%num_of_rows
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows

    flag=False

    if(cur_row==dest_row or cur_col==dest_col):#结束，先更新tensor
        flag=True
        if(cur_row==dest_row and cur_col==dest_col):#考虑到一开始两个task就被map到同一个PE的情况
            next_state_tensor=state[0]
        elif(cur_row==dest_row):
            if(cur_col<dest_col):#向East走
                for i in range(cur_col,dest_col):#更新tensor
                    next_state_tensor[0][3][cur_row*num_of_rows+i]=1
            else:#向West走
                for i in range(cur_col,dest_col,-1):#更新tensor
                    next_state_tensor[0][2][cur_row*num_of_rows+i]=1
        elif(cur_col==dest_col):
            if(cur_row<dest_row):#向South走
                for i in range(cur_row,dest_row):#更新tensor
                    next_state_tensor[0][1][i*num_of_rows+cur_col]=1
            else:#向North走
                for i in range(cur_row,dest_row,-1):
                    next_state_tensor[0][0][i*num_of_rows+cur_col]=1
    
    if(flag==False):#没有结束
        return [],0,False
    else:#结束
        #更新position
        next_position=dest_position
        #更新partRoute，此时的partRoute就是这一条链路全部的路由表，可以直接传进onlineCompute
        next_partRoute=Get_full_route_by_XY(state[2],source_position,dest_position,num_of_rows)
        #处理参数，传进onlineCompute计算pending次数
        #首先更新taskgraph里的这条链路的路由表
        for i in range(0,len(task_graph[str(task_source)]['out_links'])):
            if(int(task_graph[str(task_source)]['out_links'][i][0])==task_dest):
                task_graph[str(task_source)]['out_links'][i][2]=next_partRoute
        #处理partRoute
        partRoute_to_onlineCompute=[]
        partRoute_to_onlineCompute.append(task_source)
        partRoute_to_onlineCompute.append(task_dest)
        for i in next_partRoute:
            partRoute_to_onlineCompute.append(i)
        task=onlineTimeline("",num_of_rows)
        #print("C_fullRoute:",fullRouteFromRL)
        #print("C_partRoute:",partRoute_to_onlineCompute)
        #print("C_computing:",task_graph)
        task.loadGraphByDict(task_graph,MapResult,fullRouteFromRL,partRoute_to_onlineCompute)
        pendTimes=task.computeTime()
        #print("C_pendTimes",pendTimes)
        """
        for j in task_graph.keys():
            for k in range(0,len(task_graph[j]['out_links'])):
                task_graph[j]['out_links'][k]=task_graph[j]['out_links'][k][0:6]
        """
        #根据pendTimes计算reward
        return [next_state_tensor,next_position,next_partRoute],Get_reward_by_pendTimes(pendTimes),True


#state为[state_tensor,cur_position,partRouteFromRL]，传进来的partRoute的格式是直接的路由表，没有第一位第二位的task
def Environment(state,action,source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult):#用于获得next_state，reward，done，除了state和action，剩下的参数都是为了传进onlineCompute
    next_state_tensor=torch.Tensor(np.zeros((1,4,4*4),dtype=np.int))
    next_state_tensor.copy_(state[0])
    next_position=-1
    next_partRoute=copy.deepcopy(state[2])

    #执行action前
    cur_row=int(state[1]/num_of_rows)
    cur_col=state[1]%num_of_rows
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows
    
    #RL学习之前check一次，就能确保起码能走一步

    #开始执行action
    #state_tensor的四个channel,从0-3以此为N,S,W,E
    if(action==0):#沿x轴走
        if(cur_col<dest_col):
            next_state_tensor[0][3][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'E'])#更新路由表
            cur_col+=1#向East走了一步
        elif(cur_col>dest_col):
            next_state_tensor[0][2][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'W'])#更新路由表
            cur_col-=1#向West走了一步
    elif(action==1):#沿y轴走
        if(cur_row<dest_row):
            next_state_tensor[0][1][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'S'])#更新路由表
            cur_row+=1#向South走了一步
        elif(cur_row>dest_row):
            next_state_tensor[0][0][cur_row*num_of_rows+cur_col]=1#更新tensor
            next_partRoute.append([cur_row*num_of_rows+cur_col,'N'])#更新路由表
            cur_row-=1#向North走了一步
    
    next_position=cur_row*num_of_rows+cur_col

    ret_state,ret_reward,done=check_if_Done([next_state_tensor,next_position,next_partRoute],source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult)

    if(done==True):
        return ret_state,ret_reward,done
    else:#没有结束，需要计算reward
        #根据XY-routing补全路由表，然后传进onlineCompute
        fullRouteByXY=Get_full_route_by_XY(next_partRoute,source_position,dest_position,num_of_rows)
        #处理参数，传进onlineCompute计算pending次数
        #首先更新taskgraph里的这条链路的路由表
        for i in range(0,len(task_graph[str(task_source)]['out_links'])):
            if(int(task_graph[str(task_source)]['out_links'][i][0])==task_dest):
                task_graph[str(task_source)]['out_links'][i][2]=fullRouteByXY
        #处理partRoute
        partRoute_to_onlineCompute=[]
        partRoute_to_onlineCompute.append(task_source)
        partRoute_to_onlineCompute.append(task_dest)
        for i in next_partRoute:
            partRoute_to_onlineCompute.append(i)
        task=onlineTimeline("",num_of_rows)
        #print("fullRoute:",fullRouteFromRL)
        #print("partRoute:",partRoute_to_onlineCompute)
        #print("computing:",task_graph)
        task.loadGraphByDict(task_graph,MapResult,fullRouteFromRL,partRoute_to_onlineCompute)
        pendTimes=task.computeTime()
        #print("pendTimes",pendTimes)
        """
        for j in task_graph.keys():
            for k in range(0,len(task_graph[j]['out_links'])):
                task_graph[j]['out_links'][k]=task_graph[j]['out_links'][k][0:6]
        """
        #根据pendTimes计算reward
        return [next_state_tensor,next_position,next_partRoute],Get_reward_by_pendTimes(pendTimes),False



class ActorCritic(nn.Module):#输入：channel*length,4是channel,N是PE 输出：1*action_space和1*1
    def __init__(self, input_channel,input_length, action_space):#input_length是PE的个数
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(#输入为1*input_channel*input_length
            #nn.Linear(num_inputs, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, 1)#1*1
            nn.Conv1d(input_channel,1,input_length,1)#输出为 1*1*1
        )
        
        self.actor = nn.Sequential(
            #nn.Linear(num_inputs, hidden_size),
            #nn.ReLU(),
            #nn.Linear(hidden_size, action_space),#1*action_space
            #nn.Softmax(dim=1),
            nn.Conv1d(input_channel,1,(input_length-action_space+1),1),
            nn.Softmax(dim=2)#输出为1*1*action_space
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)#.sample()得到的是1*1的二维矩阵tensor，例如[[1]]，再.numpy后可以得到正常的矩阵
        return dist, value

# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, input_channel, input_length, action_space):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel,1,(input_length-action_space+1),1),
            nn.Softmax(dim=2)#输出为1*1*action_space
        )
    
    def forward(self, X):
        return self.model(X).squeeze(0).squeeze(0)

# Critic module
class Critic(nn.Module):
    def __init__(self, input_channel, input_length):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(input_channel,1,input_length,1)#输出为 1*1*1
        )
    
    def forward(self, X):
        return self.model(X).squeeze(0).squeeze(0)

if __name__ == '__main__':
    hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N12_autocor.tgff')
    adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
    print(adj_matrix)
    print(total_needSend)
    print(total_needReceive)
    print(execution)
    print(find_start_task(adj_matrix,num_of_tasks))
