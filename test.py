
import math
import random

#import gym  
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import time
from libs import init,Get_full_route_by_XY
from queue import Queue
from OnlineCompute1 import onlineTimeline


def Environment(state,action,source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult):#用于获得next_state，reward，done，除了state和action，剩下的参数都是为了传进onlineCompute
    next_state_tensor=state[0]
    next_position=-1
    next_partRoute=state[2]

    cur_row=int(state[1]/num_of_rows)
    cur_col=state[1]%num_of_rows
    dest_row=int(dest_position/num_of_rows)
    dest_col=dest_position%num_of_rows
    
    if(cur_row==dest_row and cur_col==dest_col):#考虑到一开始两个task就被map到同一个PE的情况，这种情况不用onlinecompute，直接结束，返回next state，routed的reward给1？这样合理吗
        #更新tensor
        next_position=dest_position
        return [next_state_tensor,next_position,[]],1,True


    if(cur_row==dest_row): #两个task走到了同一行，直接结束，需要更新state，然后将现在的state传入onlinecompute计算reward
        if(cur_col<dest_col):#向East走
            for i in range(cur_col,dest_col):#更新tensor
                next_state_tensor[0][3][cur_row*num_of_rows+i]=1
        else:#向West走
            for i in range(dest_col,cur_col):#更新tensor
                next_state_tensor[0][2][cur_row*num_of_rows+i]=1
        #更新position
        next_position=dest_position
        #更新partRoute，此时的partRoute就是这一条链路全部的路由表，可以直接传进onlineCompute
        next_partRoute=Get_full_route_by_XY(state[2],source_position,dest_position,num_of_rows)
        #处理参数，传进onlineCompute计算pending次数
        #首先更新taskgraph里的这条链路的路由表
        for i in task_graph[str(task_source)]['out_links']:
            if(int(i[0])==task_dest):
                i[2]=next_partRoute
        #处理partRoute
        partRoute_to_onlineCompute=[]
        partRoute_to_onlineCompute.append(task_source)
        partRoute_to_onlineCompute.append(task_dest)
        partRoute_to_onlineCompute.append(next_partRoute)
        task=onlineTimeline("",num_of_rows)
        task.loadGraphByDict(task_graph,MapResult,fullRouteFromRL,partRoute_to_onlineCompute)
        pendTimes=task.computeTime()
        #pendTimes=0
        #根据pendTimes计算reward
        return [next_state_tensor,next_position,next_partRoute],pendTimes,True

task_graph={'1': {'total_needSend': 48, 'out_links': [['2', 48, [], 0, 0, -1]], 'total_needReceive': 0, 'exe_time': 10}, '2': {'total_needSend': 0, 'out_links': [], 'total_needReceive': 48, 'exe_time': 10}}
MapResult=[-1,4,6,10,11]

state_tensor=torch.Tensor(np.zeros((1,4,4*4),dtype=np.int))
cur_position=4
partRoute=[]
next_state,pendTimes,done=Environment([state_tensor,cur_position,partRoute],action=0,source_position=4,dest_position=6,num_of_rows=4,task_graph=task_graph,fullRouteFromRL={},task_source=1,task_dest=2,MapResult=MapResult)
print(next_state[0])
print(next_state[1])
print(next_state[2])
print(pendTimes)
print(done)


"""
state=torch.Tensor(np.random.rand(1,1,4))
print(state)
softmax=nn.Softmax(dim=2)
probs=softmax(state)
print(probs,type(probs))
#print(probs[0][0],type(probs[0][0]))
dist=Categorical(probs)
print(dist)
action=dist.sample()
print(action,type(action))
log_prob=dist.log_prob(action)
print(log_prob)


part=[1,5,[0,"E"],[1,"S"]]
route=[[0,"E"],[0,"N"],[2,"S"],[3,"W"]]

task_graph={}
task_graph.update({"1":{"total_needSend": 2, "out_links": [["2", 2, [], 0, 0, -1]], "total_needReceive": 0, "exe_time": 1}})
#print(task_graph)
task_graph["1"]["out_links"].append(["3", 1, [[0, "S"]], 1, 2, 1])
task_graph["1"]["total_needSend"]+=1
#print(task_graph)

q=Queue(maxsize=0)
q.put(1)
q.put(2)
q.put(3)
print(q.queue)
x=q.get()

tmp={1:100,2:50,3:75}
tmp=sorted(tmp.items(), key=lambda x:x[1])
for i in tmp:
    print(i[0],type(i[0]))
"""
#hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N4_test.tgff')



