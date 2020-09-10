
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
        #根据pendTimes计算reward
        return [next_state_tensor,next_position,next_partRoute],pendTimes,True


#state为[state_tensor,cur_position,partRouteFromRL]，传进来的partRoute的格式是直接的路由表，没有第一位第二位的task
def Environment(state,action,source_position,dest_position,num_of_rows,task_graph,fullRouteFromRL,task_source,task_dest,MapResult):#用于获得next_state，reward，done，除了state和action，剩下的参数都是为了传进onlineCompute
    next_state_tensor=state[0]
    next_position=-1
    next_partRoute=state[2]

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
        for i in task_graph[str(task_source)]['out_links']:
            if(int(i[0])==task_dest):
                i[2]=fullRouteByXY
        #处理partRoute
        partRoute_to_onlineCompute=[]
        partRoute_to_onlineCompute.append(task_source)
        partRoute_to_onlineCompute.append(task_dest)
        partRoute_to_onlineCompute.append(next_partRoute)
        task=onlineTimeline("",num_of_rows)
        task.loadGraphByDict(task_graph,MapResult,fullRouteFromRL,partRoute_to_onlineCompute)
        pendTimes=task.computeTime()
        #根据pendTimes计算reward
        return [next_state_tensor,next_position,next_partRoute],pendTimes,False


task_graph={'1': {'total_needSend': 48, 'out_links': [['2', 48, [], 0, 0, -1]], 'total_needReceive': 0, 'exe_time': 10}, '2': {'total_needSend': 0, 'out_links': [], 'total_needReceive': 48, 'exe_time': 10}}
MapResult=[-1,4,6,10,11]

state_tensor=torch.Tensor(np.zeros((1,4,4*4),dtype=np.int))
cur_position=4
partRoute=[]
end_state,tmp_reward,tmp_done=check_if_Done([state_tensor,cur_position,partRoute],source_position=4,dest_position=13,num_of_rows=4,task_graph=task_graph,fullRouteFromRL={},task_source=1,task_dest=2,MapResult=MapResult)
#next_state,pendTimes,done=Environment([state_tensor,cur_position,partRoute],action=0,source_position=4,dest_position=6,num_of_rows=4,task_graph=task_graph,fullRouteFromRL={},task_source=1,task_dest=2,MapResult=MapResult)
if(tmp_done==False):
    print("exe action")
    next_state,reward,done=Environment([state_tensor,cur_position,partRoute],action=0,source_position=4,dest_position=13,num_of_rows=4,task_graph=task_graph,fullRouteFromRL={},task_source=1,task_dest=2,MapResult=MapResult)
    print(next_state[0])
    print(next_state[1])
    print(next_state[2])
    print(reward)
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

"""
#hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N4_test.tgff')



