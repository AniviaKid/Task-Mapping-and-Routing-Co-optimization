
import math
import random

#import gym  
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
import time
from libs import init,Get_full_route_by_XY,Environment,check_if_Done,Critic,Get_detailed_data,Get_rand_computation_ability2,Get_mapping_exe_time
from queue import Queue

import datetime
#from MyOnlineCompute import onlineTimeline

"""
task_graph={"1": {"total_needSend": 3, "out_links": [["2", 2, [], 0, 0, -1], ["3", 1, [[0, "S"]], 1, 2, 1]], "total_needReceive": 0, "exe_time": 1}, "2": {"total_needSend": 7, "out_links": [["4", 3, [[0, "E"]], 4, 7, 2], ["5", 4, [], 0, 0, -1]], "total_needReceive": 2, "exe_time": 3}, "3": {"total_needSend": 2, "out_links": [["6", 2, [], 0, 0, -1]], "total_needReceive": 1, "exe_time": 2}, "4": {"total_needSend": 2, "out_links": [["7", 2, [], 0, 0, -1]], "total_needReceive": 3, "exe_time": 5}, "5": {"total_needSend": 2, "out_links": [["7", 2, [[0, "E"]], 6, 8, 3]], "total_needReceive": 4, "exe_time": 2}, "6": {"total_needSend": 1, "out_links": [["7", 1, [[2, "N"], [0, "E"]], 6, 7, 4]], "total_needReceive": 2, "exe_time": 4}, "7": {"total_needSend": 0, "out_links": [], "total_needReceive": 5, "exe_time": 1}}
"""
task_graph={"1": {"total_needSend": 2, "out_links": [["2", 2, [[1,'E'],[2,'E']], 0, 0, -1]], "total_needReceive": 0, "exe_time": 1}, "2": {"total_needSend": 0, "out_links": [], "total_needReceive": 2, "exe_time": 3},"4": {"total_needSend": 2, "out_links": [["5", 2, [[2,'S'],[6,'E']], 0, 0, -1]], "total_needReceive": 0, "exe_time": 5}, "5": {"total_needSend": 0, "out_links": [], "total_needReceive": 2, "exe_time": 2}}
MapResult=[-1, 1, 3, 5, 2, 7, 0, 11]
#test=onlineTimeline("",4)
#test.loadGraphByDict(task_graph,MapResult,[],[0,0],7)
test=[('2,3',20),('1,4',50),('5,6',10)]
#test.sort(key=lambda x: x[1])
test.pop(0)
test.pop(0)
print(test)

"""
task_graph={'1': {'total_needSend': 48, 'out_links': [['2', 48, [], 0, 0, -1]], 'total_needReceive': 0, 'exe_time': 10}, '2': {'total_needSend': 0, 'out_links': [], 'total_needReceive': 48, 'exe_time': 10}}
MapResult=[-1,4,13,10,11]

state_tensor=torch.Tensor(np.zeros((1,4,4*4),dtype=np.int))
cur_position=4
partRoute=[]
state=[state_tensor,cur_position,partRoute]
end_state,tmp_reward,tmp_done=check_if_Done(state,source_position=4,dest_position=MapResult[2],num_of_rows=4,task_graph=task_graph,fullRouteFromRL={},task_source=1,task_dest=2,MapResult=MapResult)
if(tmp_done==False):
    print("exe action")
    next_state,reward,done=Environment(state,action=1,source_position=4,dest_position=MapResult[2],num_of_rows=4,task_graph=task_graph,fullRouteFromRL={},task_source=1,task_dest=2,MapResult=MapResult)
    print(state[0])
    print(state[1])
    print(state[2])
    print(next_state[0])
    print(next_state[1])
    print(next_state[2])
    state=next_state
    print(state[0])
    print(state[1])
    print(state[2])
    print(reward)
    print(done)




"""
"""
#state=torch.Tensor(np.random.rand(1,1,2))
state=torch.Tensor(np.random.rand(1,4,16))
#print(state)
softmax=nn.Softmax(dim=2)
critic=Critic(4,16)
value=critic(state)
print(value,type(value),value.shape)

probs=softmax(state)
#print(probs,type(probs),probs.shape)
#print(probs[0][0],type(probs[0][0]))
dist=Categorical(probs[0][0])
print(dist)
action=dist.sample()
print(action,type(action),action.shape)
tmp=int(action)
print("test tmp",tmp,type(tmp))
log_prob=dist.log_prob(action)
print(log_prob,log_prob.shape)
"""

"""
hyperperiod,num_of_tasks,edges,comp_cost=init('./task graph/N12_autocor_AIR1.tgff')
adj_matrix,total_needSend,total_needReceive,execution=Get_detailed_data(num_of_tasks,edges,comp_cost)
computation_ability=Get_rand_computation_ability2(num_of_rows=4)#2的指数级，4/8/16
Tasks_position_current_solution={0: 4, 1: 6, 2: 6, 3: 6, 4: 11, 5: 6, 6: 6, 7: 2, 8: 13, 9: 8, 10: 1, 11: 5}
PEs_task_current_solution=[]
for i in range(0,4*4):
    PEs_task_current_solution.append([])
for i in Tasks_position_current_solution.keys():
    PEs_task_current_solution[Tasks_position_current_solution[i]].append(i)
print(num_of_tasks)
print(adj_matrix)
print(execution)
print(computation_ability)
ret=Get_mapping_exe_time(PEs_task_current_solution,Tasks_position_current_solution,computation_ability,4,execution)
print(ret)
"""




